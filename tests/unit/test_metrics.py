#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_fit_statistics_binary(cancer_dataset):
    sklearn = pytest.importorskip('sklearn')

    from sklearn.ensemble import RandomForestClassifier
    from sasctl.utils.metrics import compare

    model = RandomForestClassifier()
    X = cancer_dataset.drop('Type', axis=1)
    y = cancer_dataset['Type']
    model.fit(X, y)

    stats = compare.fit_statistics(model, train=(X, y))

    assert isinstance(stats, dict)

    # Should only contain stats for training data
    assert len(stats['data']) == 1

    assert stats['data'][0]['rowNumber'] == 1
    assert stats['data'][0]['dataMap']['_DataRole_'] == 'TRAIN'
    assert stats['data'][0]['dataMap']['_NObs_'] == X.shape[0]
    assert stats['data'][0]['dataMap']['_DIV_'] == X.shape[0]

    for stat in ('_MCE_', ):
        assert stats['data'][0]['dataMap'][stat] is not None


def test_fit_statistics_regression(boston_dataset):
    sklearn = pytest.importorskip('sklearn')

    from sklearn.ensemble import RandomForestRegressor
    from sasctl.utils.metrics import compare

    model = RandomForestRegressor()
    X = boston_dataset.drop('Price', axis=1)
    y_true = boston_dataset['Price']
    model.fit(X, y_true)

    stats = compare.fit_statistics(model, train=(X, y_true))

    assert isinstance(stats, dict)

    # Should only contain stats for training data
    assert len(stats['data']) == 1

    assert stats['data'][0]['rowNumber'] == 1
    assert stats['data'][0]['dataMap']['_DataRole_'] == 'TRAIN'
    assert stats['data'][0]['dataMap']['_NObs_'] == X.shape[0]
    assert stats['data'][0]['dataMap']['_DIV_'] == X.shape[0]

    for stat in ('_ASE_', ):
        assert stats['data'][0]['dataMap'][stat] is not None


def test_fit_statistics_multiclass(iris_dataset):
    sklearn = pytest.importorskip('sklearn')

    from sklearn.ensemble import RandomForestClassifier
    from sasctl.utils.metrics import compare

    model = RandomForestClassifier()
    X = iris_dataset.drop('Species', axis=1)
    y_true = iris_dataset['Species']
    model.fit(X, y_true)

    stats = compare.fit_statistics(model, (X, y_true))

    assert isinstance(stats, dict)

    # Should only contain stats for training data
    assert len(stats['data']) == 1

    assert stats['data'][0]['rowNumber'] == 1
    assert stats['data'][0]['dataMap']['_DataRole_'] == 'TRAIN'
    assert stats['data'][0]['dataMap']['_NObs_'] == X.shape[0]
    assert stats['data'][0]['dataMap']['_DIV_'] == X.shape[0]

    for stat in ('_MCE_', ):
        assert stats['data'][0]['dataMap'][stat] is not None


def test_roc_statistics_binary(cancer_dataset):
    sklearn = pytest.importorskip('sklearn')

    from sklearn.ensemble import RandomForestClassifier
    from sasctl.utils.metrics import compare

    model = RandomForestClassifier()
    X = cancer_dataset.drop('Type', axis=1)
    y = cancer_dataset['Type']
    model.fit(X, y)

    stats = compare.roc_statistics(model, train=(X, y))

    assert isinstance(stats, dict)

    # Should only contain stats for training data
    assert len(stats['data']) == 3


def test_lift_statistics_binary(cancer_dataset):
    sklearn = pytest.importorskip('sklearn')

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sasctl.utils.metrics import compare

    model = RandomForestClassifier()
    X = cancer_dataset.drop('Type', axis=1)
    y = cancer_dataset['Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)

    stats = compare.lift_statistics(model, train=(X_train, y_train),
                                    test=(X_test, y_test),
                                    event='malignant')

    assert isinstance(stats, dict)

    # Should only contain stats for training data
    assert len(stats['data']) == 3