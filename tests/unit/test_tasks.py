#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_sklearn_metadata():
    pytest.importorskip('sklearn')

    from sasctl.tasks import _sklearn_to_dict
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    info = _sklearn_to_dict(LinearRegression())
    assert info['algorithm'] == 'Linear regression'
    assert info['function'] == 'Prediction'

    info = _sklearn_to_dict(LogisticRegression())
    assert info['algorithm'] == 'Logistic regression'
    assert info['function'] == 'Classification'

    info = _sklearn_to_dict(SVC())
    assert info['algorithm'] == 'Support vector machine'
    assert info['function'] == 'Classification'

    info = _sklearn_to_dict(GradientBoostingClassifier())
    assert info['algorithm'] == 'Gradient boosting'
    assert info['function'] == 'Classification'

    info = _sklearn_to_dict(DecisionTreeClassifier())
    assert info['algorithm'] == 'Decision tree'
    assert info['function'] == 'Classification'

    info = _sklearn_to_dict(RandomForestClassifier())
    assert info['algorithm'] == 'Forest'
    assert info['function'] == 'Classification'