#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_fit_statistics_binary(cancer_dataset):
    sklearn = pytest.importorskip('sklearn')

    from sklearn.ensemble import RandomForestClassifier
    from sasctl.utils import metrics

    model = RandomForestClassifier()
    X = cancer_dataset.drop('Type', axis=1)
    y = cancer_dataset['Type']
    model.fit(X, y)

    stats = metrics.fit_statistics(model, train=(X, y))

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
    from sasctl.utils import metrics

    model = RandomForestRegressor()
    X = boston_dataset.drop('Price', axis=1)
    y_true = boston_dataset['Price']
    model.fit(X, y_true)

    stats = metrics.fit_statistics(model, train=(X, y_true))

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
    from sasctl.utils import metrics

    model = RandomForestClassifier()
    X = iris_dataset.drop('Species', axis=1)
    y_true = iris_dataset['Species']
    model.fit(X, y_true)

    stats = metrics.fit_statistics(model, (X, y_true))

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
    from sasctl.utils import metrics

    model = RandomForestClassifier()
    X = cancer_dataset.drop('Type', axis=1)
    y = cancer_dataset['Type']
    model.fit(X, y)

    stats = metrics.roc_statistics(model, train=(X, y))

    assert isinstance(stats, dict)

    # Should only contain stats for training data
    assert len(stats['data']) == 3


def test_lift_statistics_binary(cancer_dataset):
    sklearn = pytest.importorskip('sklearn')
    pd = pytest.importorskip('pandas')
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sasctl.utils import metrics

    TARGET = 'Type'

    model = DecisionTreeClassifier()
    X = cancer_dataset.drop(TARGET, axis=1)
    y = cancer_dataset[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)

    stats = metrics.lift_statistics(model, train=(X_train, y_train),
                                    test=(X_test, y_test),
                                    event='malignant')

    assert isinstance(stats, dict)
    assert stats['name'] == 'dmcas_lift'
    assert 'parameterMap' in stats

    data = stats['data']
    assert isinstance(data, list)

    df = pd.DataFrame((row['dataMap'] for row in data))
    rownums = pd.Series((row['rowNumber'] for row in data))
    df = pd.concat([df, rownums], axis=1)

    # Verify data set names
    assert df._DataRole_.isin(('TRAIN', 'TEST')).all()

    # Each row should refer to the target variable
    assert (df._Column_ == TARGET).all()

    # "_CumLift_" should never drop below 1
    min_cum_lift = round(df.groupby('_DataRole_')._CumLift_.min(), 10)
    assert (min_cum_lift >= 1).all()

    # "_Lift_" should never drop below 0
    min_lift = round(df.groupby('_DataRole_')._Lift_.min(), 10)
    assert (min_lift >= 0).all()

    # Number of observations across all groups should match
    num_obs = df.groupby('_DataRole_')._NObs_.sum()
    assert num_obs['TEST'] == len(X_test)
    assert num_obs['TRAIN'] == len(X_train)
