#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os

import pytest
from six.moves import mock

from sasctl.utils.pyml2ds import pyml2ds


dirname = os.path.dirname
DATA_PATH = os.path.join(dirname(dirname(__file__)), 'pyml2ds_data')


def test_xgb2ds(tmpdir):
    pytest.importorskip('xgboost')

    IN_PKL = os.path.join(DATA_PATH, 'xgb.pkl')
    OUT_SAS = os.path.join(str(tmpdir), 'xgb.sas')
    EXPECTED_SAS = os.path.join(DATA_PATH, 'xgb.sas')

    from sasctl.utils.pyml2ds.connectors.ensembles.xgb import XgbTreeParser

    # Expected output contains integer values instead of floats.
    # Convert to ensure match.
    class TestXgbTreeParser(XgbTreeParser):
        def _split_value(self):
            val = super(TestXgbTreeParser, self)._split_value()
            return int(float(val))

        def _leaf_value(self):
            val = super(TestXgbTreeParser, self)._leaf_value()
            return int(float(val))

    test_parser = TestXgbTreeParser()

    with mock.patch('sasctl.utils.pyml2ds.connectors.ensembles.xgb.XgbTreeParser') as parser:
        parser.return_value = test_parser
        pyml2ds(IN_PKL, OUT_SAS)

    result = open(OUT_SAS, 'rb').read()
    expected = open(EXPECTED_SAS, 'rb').read()
    assert result == expected


def test_lgb2ds(tmpdir):
    pytest.importorskip('lightgbm')

    IN_PKL = os.path.join(DATA_PATH, 'lgb.pkl')
    OUT_SAS = os.path.join(str(tmpdir), 'lgb.sas')
    EXPECTED_SAS = os.path.join(DATA_PATH, 'lgb.sas')

    from sasctl.utils.pyml2ds.connectors.ensembles.lgb import LightgbmTreeParser

    # Expected output contains integer values instead of floats.
    # Convert to ensure match.
    class TestLightgbmTreeParser(LightgbmTreeParser):
        def _split_value(self):
            val = super(TestLightgbmTreeParser, self)._split_value()
            return int(float(val))

        def _leaf_value(self):
            val = super(TestLightgbmTreeParser, self)._leaf_value()
            return int(float(val))

    test_parser = TestLightgbmTreeParser()

    with mock.patch('sasctl.utils.pyml2ds.connectors.ensembles.lgb.LightgbmTreeParser') as parser:
        parser.return_value = test_parser
        pyml2ds(IN_PKL, OUT_SAS)


    result = open(OUT_SAS, 'rb').read()
    expected = open(EXPECTED_SAS, 'rb').read()
    assert result == expected


def test_gbm2ds(tmpdir):
    IN_PKL = os.path.join(DATA_PATH, 'gbm.pmml')
    OUT_SAS = os.path.join(str(tmpdir), 'gbm.sas')
    EXPECTED_SAS = os.path.join(DATA_PATH, 'gbm.sas')

    from sasctl.utils.pyml2ds.connectors.ensembles.pmml import PmmlTreeParser

    # Expected output contains integer values instead of floats.
    # Convert to ensure match.
    class TestPmmlTreeParser(PmmlTreeParser):
        def _split_value(self):
            val = super(TestPmmlTreeParser, self)._split_value()
            return int(float(val))

        def _leaf_value(self):
            val = super(TestPmmlTreeParser, self)._leaf_value()
            return int(float(val))

    test_parser = TestPmmlTreeParser()

    with mock.patch('sasctl.utils.pyml2ds.connectors.ensembles.pmml.PmmlTreeParser') as parser:
        parser.return_value = test_parser
        pyml2ds(IN_PKL, OUT_SAS)

    result = open(OUT_SAS, 'rb').read()
    expected = open(EXPECTED_SAS, 'rb').read()
    assert result == expected


