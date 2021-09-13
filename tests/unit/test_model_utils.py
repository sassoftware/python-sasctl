#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
# test base

# test scikit

# pytorch?  tf?  statsmodel?

from sasctl.utils.models import ModelInfo


def test_scikit_model_properties():
    pytest.importorskip('sklearn')
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sasctl.utils.models import ScikitModelInfo

    lin_reg_info = ModelInfo(LinearRegression())
    assert isinstance(lin_reg_info, ScikitModelInfo)
    assert lin_reg_info.tool == 'scikit'
    assert lin_reg_info.function == 'prediction'
    assert lin_reg_info.function_names == ['predict']
    assert lin_reg_info.algorithm == 'Linear regression'

    log_reg_info = ModelInfo(LogisticRegression())
    assert isinstance(log_reg_info, ScikitModelInfo)
    assert log_reg_info.tool == 'scikit'
    assert log_reg_info.function == 'classification'
    assert log_reg_info.function_names == ['predict', 'predict_proba']
    assert log_reg_info.algorithm == 'Logistic regression'
