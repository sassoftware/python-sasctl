#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from six.moves import mock

from sasctl.utils import metrics


def test_fit_statistics_binary_pandas(cancer_dataset):
    """Check fit statistics for a binary classification model with Pandas inputs."""
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
    datamap = stats['data'][0]['dataMap']

    assert datamap['_DataRole_'] == 'TRAIN'
    assert datamap['_NObs_'] == X.shape[0]
    assert datamap['_DIV_'] == X.shape[0]

    assert datamap['_ASE_'] is not None
    assert datamap['_C_'] is not None
    assert datamap['_GAMMA_'] is None
    assert datamap['_GINI_'] is not None
    assert datamap['_KS_'] is not None
    assert datamap['_MCE_'] is not None
    assert datamap['_MCLL_'] is not None
    assert datamap['_RASE_'] is not None
    assert datamap['_TAU_'] is None


def test_fit_statistics_binary_numpy(cancer_dataset):
    """Check fit statistics for a binary classification model with Numpy inputs."""
    sklearn = pytest.importorskip('sklearn')

    from sklearn.ensemble import RandomForestClassifier
    from sasctl.utils import metrics

    model = RandomForestClassifier()
    X = cancer_dataset.drop('Type', axis=1).values
    y = cancer_dataset['Type'].values
    model.fit(X, y)

    stats = metrics.fit_statistics(model, train=(X, y))

    assert isinstance(stats, dict)

    # Should only contain stats for training data
    assert len(stats['data']) == 1

    assert stats['data'][0]['rowNumber'] == 1
    datamap = stats['data'][0]['dataMap']

    assert datamap['_DataRole_'] == 'TRAIN'
    assert datamap['_NObs_'] == X.shape[0]
    assert datamap['_DIV_'] == X.shape[0]

    assert datamap['_ASE_'] is not None
    assert datamap['_C_'] is not None
    assert datamap['_GINI_'] is not None
    assert datamap['_KS_'] is not None
    assert datamap['_MCE_'] is not None
    assert datamap['_MCLL_'] is not None
    assert datamap['_RASE_'] is not None


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
    datamap = stats['data'][0]['dataMap']

    assert datamap['_DataRole_'] == 'TRAIN'
    assert datamap['_NObs_'] == X.shape[0]
    assert datamap['_DIV_'] == X.shape[0]

    assert datamap['_ASE_'] is not None
    assert datamap['_RASE_'] is not None


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
    datamap = stats['data'][0]['dataMap']

    assert datamap['_DataRole_'] == 'TRAIN'
    assert datamap['_NObs_'] == X.shape[0]
    assert datamap['_DIV_'] == X.shape[0]

    assert datamap['_ASE_'] is None
    assert datamap['_C_'] is None
    assert datamap['_GAMMA_'] is None
    assert datamap['_GINI_'] is None
    assert datamap['_KS_'] is None
    assert datamap['_MCE_'] is None
    assert datamap['_MCLL_'] is None
    assert datamap['_RASE_'] is None
    assert datamap['_TAU_'] is None


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
    assert df.shape[0] == len(data)

    # "_CumLift_" should never drop below 1
    assert 'TRAIN' in df._DataRole_.values
    assert 'TEST' in df._DataRole_.values
    assert round(df._CumLift_.min(), 10) >= 1
    # min_cum_lift = round(df.groupby('_DataRole_')._CumLift_.min(), 10)
    # assert (min_cum_lift >= 1).all()

    # "_Lift_" should never drop below 0
    assert round(df._Lift_.min(), 10) >= 0
    # min_lift = round(df.groupby('_DataRole_')._Lift_.min(), 10)
    # assert (min_lift >= 0).all()

    # Number of observations across all groups should match
    num_obs = df.groupby('_DataRole_')._NObs_.sum()
    assert num_obs['TEST'] == len(X_test)
    assert num_obs['TRAIN'] == len(X_train)


def test_regression_fit_stats_match_sas():
    """Ensure fit statistics for a regression model match values produced by SAS."""
    pd = pytest.importorskip('pandas')
    sklearn = pytest.importorskip('sklearn')

    from sklearn.linear_model import LinearRegression

    # NOTE: the following SAS code can be used to generate the target values.
    """
    libname mycas cas;
    
    data mycas.score;
       input good p_good;
       datalines;
    0.8224 0.7590
    0.6538 0.4632
    0.7693 0.7069
    0.7491 0.7087
    0.7779 0.7209
    0.7161 0.8389
    0.6779 0.6209
    0.6392 0.6077
    0.8090 0.9096
    0.6064 0.7355
    ;
    
    proc assess data=mycas.score nbins=2;
       var p_good;
       target good;
    run;
"""

    df = pd.DataFrame([(0.8224, 0.7590),
                       (0.6538, 0.4632),
                       (0.7693, 0.7069),
                       (0.7491, 0.7087),
                       (0.7779, 0.7209),
                       (0.7161, 0.8389),
                       (0.6779, 0.6209),
                       (0.6392, 0.6077),
                       (0.8090, 0.9096),
                       (0.6064, 0.7355)],
                      columns=['true', 'predicted'])

    mock_model = mock.Mock(spec=LinearRegression)
    mock_model.predict.return_value = df['predicted']
    dummy_input = [None] * len(df)
    results = metrics.fit_statistics(mock_model, train=(dummy_input, df['true']))

    result = results['data'][0]['dataMap']
    assert result['_DataRole_'] == 'TRAIN'
    assert result['_NObs_'] == len(df)

    assert round(result['_ASE_'], 5) == 0.00952
    assert round(result['_RASE_'], 5) == 0.09759

    # Not populated by Model Studio for regression models
    assert result['_MCE_'] is None
    assert result['_KS_'] is None
    assert result['_KSCut_'] is None
    assert result['_KSPostCutoff_'] is None
    assert result['_C_'] is None
    assert result['_GAMMA_'] is None
    assert result['_GINI_'] is None
    assert result['_TAU_'] is None
    assert result['_MCLL_'] is None


def test_binary_fit_stats_match_sas():
    """Ensure fit statistics for a binary classification model match values produced by SAS."""
    pd = pytest.importorskip('pandas')
    sklearn = pytest.importorskip('sklearn')

    from sklearn.linear_model import LogisticRegression

    # NOTE: the following SAS code can be used to generate the target values.
    """
    libname mycas cas;

    data mycas.score2;
       length good_bad $4;
       input _PartInd_ good_bad p_good p_bad;
       datalines;
    0 good 0.6675 0.3325
    0 good 0.5189 0.4811
    0 good 0.6852 0.3148
    0 bad  0.0615 0.9385
    0 bad  0.3053 0.6947
    0 bad  0.6684 0.3316
    0 good 0.6422 0.3578
    0 good 0.6752 0.3248
    0 good 0.5396 0.4604
    0 good 0.4983 0.5017
    0 bad  0.1916 0.8084
    0 good 0.5722 0.4278
    0 good 0.7099 0.2901
    0 good 0.4642 0.5358
    0 good 0.4863 0.5137
    ;

    proc assess data=mycas.score2 ncuts=4 nbins=5;
       var p_good;
       target good_bad / event="good" level=nominal;
       fitstat pvar=p_bad / pevent="bad" ;
    run;
    """

    df = pd.DataFrame([('good', 0.6675, 0.3325),
                       ('good', 0.5189, 0.4811),
                        ('good', 0.6852, 0.3148),
                        ('bad',  0.0615, 0.9385),
                        ('bad',  0.3053, 0.6947),
                        ('bad',  0.6684, 0.3316),
                        ('good', 0.6422, 0.3578),
                        ('good', 0.6752, 0.3248),
                        ('good', 0.5396, 0.4604),
                        ('good', 0.4983, 0.5017),
                        ('bad',  0.1916, 0.8084),
                        ('good', 0.5722, 0.4278),
                        ('good', 0.7099, 0.2901),
                        ('good', 0.4642, 0.5358),
                        ('good', 0.4863, 0.5137)],
                      columns=['true', 'p_good', 'p_bad'])
    df['predicted'] = 'bad'
    df.loc[df['p_good'] > 0.5, 'predicted'] = 'good'

    mock_model = mock.Mock(spec=LogisticRegression)
    mock_model.classes_ = ['bad', 'good']
    mock_model.predict.return_value = df['predicted']
    mock_model.predict_proba.return_value = df[['p_bad', 'p_good']]
    dummy_input = [None] * len(df)
    results = metrics.fit_statistics(mock_model, train=(dummy_input, df['true']))

    result = results['data'][0]['dataMap']
    assert result['_DataRole_'] == 'TRAIN'
    assert result['_NObs_'] == len(df)

    # NOTE: AUC and Gini do not exactly match SAS output since number sample points for ROC are different.
    assert round(result['_ASE_'], 5) == 0.16913
    assert round(result['_RASE_'], 5) == 0.41125
    assert round(result['_MCE_'], 5) == 0.26667
    assert round(result['_KS_'], 5) == 0.47727
    assert round(result['_C_'], 5) == 0.81818
    assert round(result['_GINI_'], 5) == 0.63636
    assert round(result['_MCLL_'], 5) == 0.51473

    # assert round(result['_GAMMA_'], 5) == ?
    # assert round(result['_TAU_'], 5) == ?


def test_multiclass_fit_stats_match_sas():
    """Ensure fit statistics for a multiclass classification model match values produced by SAS."""
    pd = pytest.importorskip('pandas')
    sklearn = pytest.importorskip('sklearn')

    from sklearn.ensemble import GradientBoostingClassifier

    # NOTE: the following SAS code can be used to generate the target values.
    """
    libname mycas cas;

    data mycas.score2;
       length good_bad $7;
       input good_bad p_good p_bad p_neutral;
       datalines;
    good 0.6675 0.3325 0
    good 0.5189 0.4811 0
    good 0.6852 0.3148 0
    bad 0.0515 0.9385 0.01
    bad 0.3053 0.5947 0.1
    bad 0.5684 0.3316 0.1
    good 0.6422 0.2578 0.1
    neutral 0.4752 0.1 0.4248
    good 0.5396 0.3604 0.1
    good 0.4983 0.5017 0.0
    bad 0.1916 0.8084 0.0
    good 0.5722 0.3278 0.1
    good 0.7099 0.1901 0.1
    neutral 0.4342 0.1 0.5358
    neutral 0.2432 0.2431 0.5137
    ;

    proc assess data=mycas.score2 ncuts=4 nbins=5;
       var p_good;
       target good_bad / event="good" level=nominal;
       fitstat pvar=p_bad p_neutral / pevent="bad neutral" ;
    run;
    """

    df = pd.DataFrame([('good', 0.6675, 0.3325, 0),
                       ('good', 0.5189, 0.4811, 0),
                       ('good', 0.6852, 0.3148, 0),
                       ('bad', 0.0515, 0.9385, 0.01),
                       ('bad', 0.3053, 0.5947, 0.1),
                       ('bad', 0.5684, 0.3316, 0.1),
                       ('good', 0.6422, 0.2578, 0.1),
                       ('neutral', 0.4752, 0.1, 0.4248),
                       ('good', 0.5396, 0.3604, 0.1),
                       ('good', 0.4983, 0.5017, 0.0),
                       ('bad', 0.1916, 0.8084, 0.0),
                       ('good', 0.5722, 0.3278, 0.1),
                       ('good', 0.7099, 0.1901, 0.1),
                       ('neutral', 0.4342, 0.1, 0.5358),
                       ('neutral', 0.2432, 0.2431, 0.5137)],
                      columns=['true', 'p_good', 'p_bad', 'p_neutral'])

    # NOTE: SAS PROC ASSESS uses probability > cutoff, not max(probability)
    # Get column name where probability > 0.5
    df['predicted'] = (df[['p_bad', 'p_neutral', 'p_good']] > 0.5).idxmax(axis=1).str.replace('p_', '')

    mock_model = mock.Mock(spec=GradientBoostingClassifier)
    mock_model.classes_ = ['bad', 'neutral', 'good']
    mock_model.predict.return_value = df['predicted']
    mock_model.predict_proba.return_value = df[['p_bad', 'p_neutral', 'p_good']]
    dummy_input = [None] * len(df)
    results = metrics.fit_statistics(mock_model, train=(dummy_input, df['true']), event='good')

    result = results['data'][0]['dataMap']
    assert result['_DataRole_'] == 'TRAIN'
    assert result['_NObs_'] == len(df)

    # Any values that haven't been matched to SAS should not be reported
    assert result['_ASE_'] is None
    assert result['_RASE_'] is None
    assert result['_MCE_'] is None
    assert result['_KS_'] is None
    assert result['_KSCut_'] is None
    assert result['_KSPostCutoff_'] is None
    assert result['_C_'] is None
    assert result['_GAMMA_'] is None
    assert result['_GINI_'] is None
    assert result['_TAU_'] is None
    assert result['_MCLL_'] is None
