#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re


def test_list_packages():
    from sasctl.utils.misc import installed_packages
    packages = installed_packages()

    # We know that these packages should always be present
    assert any(re.match('requests==.*', p) for p in packages)
    assert any(re.match('sasctl.*', p) for p in packages)  # sasctl may be installed from disk so no '=='
    assert any(re.match('pytest==.*', p) for p in packages)