#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import RestObj
from sasctl.services import relationships as rel

pytestmark = pytest.mark.usefixtures("session")


def test_list_relationships():
    relationships = rel.list_relationships()
    assert all(isinstance(f, RestObj) for f in relationships)
