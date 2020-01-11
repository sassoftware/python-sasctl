#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import six



# general additive model  (gamSelect)
# clustering        (fastKnn)
# gb
# tree
# svdd?
# deep neural
# bayes net
# factmac
# forecast
# text?

from sasctl.utils.astore import _get_model_properties


@pytest.fixture
def boston_dataset():
    """Regression dataset."""
    import pandas as pd
    from sklearn import datasets

    raw = datasets.load_boston()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df['Price'] = raw.target
    return df

@pytest.fixture
def cancer_dataset():
    """Binary classification dataset."""
    import pandas as pd
    from sklearn import datasets

    raw = datasets.load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df['Type'] = raw.target
    df.Type = df.Type.astype('category')
    df.Type.cat.categories = raw.target_names
    return df


@pytest.fixture
def iris_dataset():
    """Multi-class classification dataset."""
    import pandas as pd
    from sklearn import datasets

    raw = datasets.load_iris()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df['Species'] = raw.target
    df.Species = df.Species.astype('category')
    df.Species.cat.categories = raw.target_names
    return df


def test_glm(cas_session, boston_dataset):
    target = {
        'tool': 'SAS Visual Analytics',
        'targetVariable': 'Price',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Prediction',
        'algorithm': 'Linear regression'
    }

    cas_session.loadactionset('regression')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(boston_dataset).casTable

    tbl.regression.glm(target='Price',
                       inputs=list(boston_dataset.columns[:-1]),
                       savestate='astore')

    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v


def test_logistic(cas_session, iris_dataset):
    target = {
        'tool': 'SAS Visual Analytics',
        'targetVariable': 'Species',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Classification',
        'algorithm': 'Logistic regression'
    }

    cas_session.loadactionset('regression')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(iris_dataset).casTable

    tbl.regression.logistic(target='Species',
                                inputs=list(iris_dataset.columns[:-1]),
                                savestate='astore')

    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v


def test_dtree_regression(cas_session, boston_dataset):
    target = {
        'tool': 'SAS Visual Analytics',
        'targetVariable': 'Price',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Prediction',
        'algorithm': 'Decision tree'
    }

    cas_session.loadactionset('decisiontree')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(boston_dataset).casTable

    tbl.decisiontree.dtreetrain(target='Price',
                                inputs=list(boston_dataset.columns[:-1]),
                                casout='tree')

    pytest.skip('Implement.  How to get an astore?')

    cas_session.decisiontree.dtreeExportModel(modelTable='tree',
                                         casout='astore')

    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v


def test_forest_classification(cas_session, iris_dataset):
    target = {
        'tool': 'SAS Visual Data Mining and Machine Learning',
        'targetVariable': 'Species',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Classification',
        'algorithm': 'Random forest'
    }

    cas_session.loadactionset('decisiontree')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(iris_dataset).casTable

    tbl.decisiontree.foresttrain(target='Species',
                                 inputs=list(iris_dataset.columns[:-1]),
                                 saveState='astore')

    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v


def test_forest_regression(cas_session, boston_dataset):
    target = {
        'tool': 'SAS Visual Data Mining and Machine Learning',
        'targetVariable': 'Price',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Prediction',
        'algorithm': 'Random forest'
    }

    cas_session.loadactionset('decisiontree')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(boston_dataset).casTable

    tbl.decisiontree.foresttrain(target='Price',
                                 inputs=list(boston_dataset.columns[:-1]),
                                 saveState='astore')

    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v

def test_gradboost_classification(cas_session, iris_dataset):
    target = {
        'tool': 'SAS Visual Data Mining and Machine Learning',
        'targetVariable': 'Species',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Classification',
        'algorithm': 'Gradient boosting'
    }

    cas_session.loadactionset('decisiontree')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(iris_dataset).casTable

    tbl.decisiontree.gbtreetrain(target='Species',
                                inputs=list(iris_dataset.columns[:-1]),
                                savestate='astore')

    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v


def test_gradboost_regression(cas_session, boston_dataset):
    target = {
        'tool': 'SAS Visual Data Mining and Machine Learning',
        'targetVariable': 'Price',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Prediction',
        'algorithm': 'Gradient boosting'
    }

    cas_session.loadactionset('decisiontree')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(boston_dataset).casTable

    tbl.decisiontree.gbtreetrain(target='Price',
                                 inputs=list(boston_dataset.columns[:-1]),
                                 savestate='astore')

    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v


def test_neuralnet_regression(cas_session, boston_dataset):
    target = {
        'tool': 'SAS Visual Data Mining and Machine Learning',
        'targetVariable': 'Price',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Prediction',
        'algorithm': 'Neural network'
    }

    cas_session.loadactionset('neuralnet')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(boston_dataset).casTable

    tbl.neuralNet.annTrain(target='Price',
                           inputs=list(boston_dataset.columns[:-1]),
                           # modelTable='network',
                           arch='MLP',
                           hiddens=[2],
                           combs=['linear'],
                           casout='network')
                           # savestate='astore')

    pytest.skip('Implement.  How to get an astore?')
    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v


def test_svm_classification(cas_session, cancer_dataset):
    target = {
        'tool': 'SAS Visual Data Mining and Machine Learning',
        'targetVariable': 'Type',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Classification',
        'algorithm': 'Support vector machine'
    }

    cas_session.loadactionset('svm')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(cancer_dataset).casTable

    tbl.svm.svmTrain(target='Type',
                     inputs=list(cancer_dataset.columns[:-1]),
                     saveState='astore')

    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v


def test_svm_regression(cas_session, boston_dataset):
    target = {
        'tool': 'SAS Visual Data Mining and Machine Learning',
        'targetVariable': 'Price',
        'scoreCodeType': 'ds2MultiType',
        'function': 'Prediction',
        'algorithm': 'Support vector machine'
    }

    cas_session.loadactionset('svm')
    cas_session.loadactionset('astore')

    tbl = cas_session.upload(boston_dataset).casTable

    tbl.svm.svmTrain(target='Price',
                     inputs=list(boston_dataset.columns[:-1]),
                     saveState='astore')

    desc = cas_session.astore.describe(rstore='astore', epcode=True)
    props = _get_model_properties(desc)

    for k, v in six.iteritems(target):
        assert props[k] == v