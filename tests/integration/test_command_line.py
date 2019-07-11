#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl.utils.cli import main


pytestmark = pytest.mark.usefixtures('session')


def test_insecure_connection():
    """Verify that an insecure connection flag works."""

    r = main(['-k', 'folders', 'get', 'dummy_folder'])
    assert r is None


def test_secure_connection():
    """Verify that a secure connection flag works."""

    r = main(['folders', 'get', 'dummy_folder'])
    assert r is None
