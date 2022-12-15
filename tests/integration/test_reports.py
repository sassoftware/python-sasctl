#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import RestObj
from sasctl.services import reports

pytestmark = pytest.mark.usefixtures("session")


def test_list_reports():
    all_reports = reports.list_reports()

    assert len(all_reports) > 0
    assert all(isinstance(x, RestObj) for x in all_reports)
    assert all(x.type == "report" for x in all_reports)


def test_get_report():
    NAME = "User Activity"
    r = reports.get_report(NAME)

    assert isinstance(r, RestObj)
    assert r.name == NAME
