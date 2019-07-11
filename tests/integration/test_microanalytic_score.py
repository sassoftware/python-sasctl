#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.usefixtures('session')


def test_publish_python_module(caplog):
    from sasctl.services.microanalytic_score import create_module
    import logging
    caplog.set_level(logging.DEBUG, logger='sasctl.core')

    source = '\n'.join(("def myfunction(var1, var2):",
                        "    'Output: out1, out2'",
                        "    out1 = var1 + 5",
                        "    out2 = var2.upper()",
                        "    return out1, out2"))

    r = create_module(source=source, name='sasctl_testmethod')
    assert 'sasctl_testmethod' == r.id
    assert 'public' == r.scope

def test_call_python_module_steps():
    from sasctl.services.microanalytic_score import define_steps

    r = define_steps('sasctl_testmethod')
    assert (6, 'TEST') == r.myfunction(1, 'test')

