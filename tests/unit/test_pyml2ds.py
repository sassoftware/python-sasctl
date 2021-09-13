#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
from unittest import mock

import pytest

from sasctl.utils.pyml2ds import pyml2ds


dirname = os.path.dirname
DATA_PATH = os.path.join(dirname(dirname(__file__)), 'pyml2ds_data')


@pytest.mark.skip(
    'Pickle no longer loads with latest version of sklearn.  Rework to build model instead of loading.'
)
def test_xgb2ds():
    pytest.importorskip('xgboost')

    IN_PKL = os.path.join(DATA_PATH, 'xgb.pkl')
    EXPECTED_SAS = os.path.join(DATA_PATH, 'xgb_datastep')

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

    with mock.patch(
        'sasctl.utils.pyml2ds.connectors.ensembles.xgb.XgbTreeParser'
    ) as parser:
        parser.return_value = test_parser
        result = pyml2ds(IN_PKL)

    with open(EXPECTED_SAS, 'r') as f:
        expected = f.read()

    assert result == expected


def test_lgb2ds():
    pytest.importorskip('lightgbm')

    IN_PKL = os.path.join(DATA_PATH, 'lgb.pkl')
    EXPECTED_SAS = os.path.join(DATA_PATH, 'lgb_datastep')

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

    with mock.patch(
        'sasctl.utils.pyml2ds.connectors.ensembles.lgb.LightgbmTreeParser'
    ) as parser:
        parser.return_value = test_parser
        result = pyml2ds(IN_PKL)

    with open(EXPECTED_SAS, 'r') as f:
        expected = f.read()
    assert result == expected


def test_gbm2ds():
    IN_PKL = os.path.join(DATA_PATH, 'gbm.pmml')
    EXPECTED_SAS = os.path.join(DATA_PATH, 'gbm_datastep')

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

    with mock.patch(
        'sasctl.utils.pyml2ds.connectors.ensembles.pmml.PmmlTreeParser'
    ) as parser:
        parser.return_value = test_parser
        result = pyml2ds(IN_PKL)

    with open(EXPECTED_SAS, 'r') as f:
        expected = f.read()
    assert result == expected


def test_path_input(tmpdir_factory):
    """pyml2ds should accept a file path (str) as input."""
    import pickle
    from sasctl.utils.pyml2ds import pyml2ds

    # The target "model" to use
    target = {'msg': 'hello world'}

    # Pickle the "model" to a file
    temp_dir = tmpdir_factory.mktemp('pyml2ds')
    in_file = str(temp_dir.join('model.pkl'))
    out_file = str(temp_dir.join('model.sas'))
    with open(in_file, 'wb') as f:
        pickle.dump(target, f)

    with mock.patch('sasctl.utils.pyml2ds.core._check_type') as check:
        check.translate.return_value = 'translated'
        pyml2ds(in_file, out_file)

    # Verify _check_type should have been called with the "model"
    assert check.call_count == 1
    assert check.call_args[0][0] == target


def test_file_input():
    """pyml2ds should accept a file-like obj as input."""
    import io
    import pickle
    from sasctl.utils.pyml2ds import pyml2ds

    # The target "model" to use
    target = {'msg': 'hello world'}

    # Pickle the "model" to a file-like object
    in_file = io.BytesIO(pickle.dumps(target))
    out_file = 'model.sas'

    with mock.patch('sasctl.utils.pyml2ds.core._check_type') as check:
        check.translate.return_value = 'translated'
        pyml2ds(in_file, out_file)

    # Verify _check_type should have been called with the "model"
    assert check.call_count == 1
    assert check.call_args[0][0] == target


def test_pickle_input():
    """pyml2ds should accept a binary pickle string as input."""
    import pickle
    from sasctl.utils.pyml2ds import pyml2ds

    # The target "model" to use
    target = {'msg': 'hello world'}

    # Pickle the "model" to a file-like object
    in_file = pickle.dumps(target)
    out_file = 'model.sas'

    with mock.patch('sasctl.utils.pyml2ds.core._check_type') as check:
        check.translate.return_value = 'translated'
        pyml2ds(in_file, out_file)

    # Verify _check_type should have been called with the "model"
    assert check.call_count == 1
    assert check.call_args[0][0] == target
