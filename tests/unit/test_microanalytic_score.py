#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from six.moves import mock

from sasctl.services import microanalytic_score as mas

from sasctl import current_session

with mock.patch('sasctl.core.requests.Session.request'):
    current_session('example.com', 'username', 'password')


def test_create_python_module():
    with mock.patch('sasctl.services.microanalytic_score.post') as post:
        with pytest.raises(ValueError):
            mas.create_module()     # Source code is required

    with mock.patch('sasctl.services.microanalytic_score.post') as post:
        source = '\n'.join(("def testMethod(var1, var2):",
                            "    'Output: out1, out2'",
                            "    out1 = var1 + 5",
                            "    out2 = var2.upper()",
                            "    return out1, out2"))
        mas.create_module(source=source)

        assert post.call_count == 1
        json = post.call_args[1].get('json', {})
        assert 'text/x-python' == json['type']
        assert 'public' == json['scope']


def test_delete_module(caplog):
    import logging
    caplog.set_level(logging.INFO, 'sasctl._services.service')

    # Delete should succeed even if object couldn't be found on server
    with mock.patch('sasctl._services.microanalytic_score.MicroAnalyticScore'
                    '.get') as get:
        get.return_value = None

        assert mas.delete_module('spam') is None
        assert any("Object 'spam' not found" in r.msg for r in caplog.records)

