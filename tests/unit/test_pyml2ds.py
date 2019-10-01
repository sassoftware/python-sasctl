#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
dirname = os.path.dirname

import pytest
from sasctl.utils.pyml2ds import pyml2ds


DATA_PATH = os.path.join(dirname(dirname(__file__)), 'pyml2ds_data')


def test_xgb2ds(tmpdir):
    IN_PKL = os.path.join(DATA_PATH, 'xgb.pkl')
    OUT_SAS = os.path.join(tmpdir, 'xgb.sas')
    EXPECTED_SAS = os.path.join(DATA_PATH, 'xgb.sas')

    pyml2ds(IN_PKL, OUT_SAS)
    result = open(OUT_SAS, 'rb').read()
    expected = open(EXPECTED_SAS, 'rb').read()
    assert result == expected


def test_lgb2ds(tmpdir):
    IN_PKL = os.path.join(DATA_PATH, 'lgb.pkl')
    OUT_SAS = os.path.join(tmpdir, 'lgb.sas')
    EXPECTED_SAS = os.path.join(DATA_PATH, 'lgb.sas')

    pyml2ds(IN_PKL, OUT_SAS)
    result = open(OUT_SAS, 'rb').read()
    expected = open(EXPECTED_SAS, 'rb').read()
    assert result == expected


def test_gbm2ds(tmpdir):
    IN_PKL = os.path.join(DATA_PATH, 'gbm.pmml')
    OUT_SAS = os.path.join(tmpdir, 'gbm.sas')
    EXPECTED_SAS = os.path.join(DATA_PATH, 'gbm.sas')

    pyml2ds(IN_PKL, OUT_SAS)
    result = open(OUT_SAS, 'rb').read()
    expected = open(EXPECTED_SAS, 'rb').read()
    assert result == expected
