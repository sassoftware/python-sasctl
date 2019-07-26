#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import warnings

import pytest


# Every test function in the module will automatically receive the session fixture
pytestmark = pytest.mark.usefixtures('session')

ASTORE_MODEL_NAME = 'Astore Model'
SCIKIT_MODEL_NAME = 'Scikit Model'
PROJECT_NAME = 'Test Project'


@pytest.fixture
def sklearn_model():
    """Returns a simple Scikit-Learn model """

    try:
        import pandas as pd
    except ImportError:
        pytest.skip('Package `pandas` not found.')

    try:
        from sklearn import datasets
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip('Package `sklearn` not found.')

    raw = datasets.load_iris()
    iris = pd.DataFrame(raw.data, columns=raw.feature_names)
    iris = iris.join(pd.DataFrame(raw.target))
    iris.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    iris['Species'] = iris['Species'].astype('category')
    iris.Species.cat.categories = raw.target_names

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(iris.iloc[:, 0:4], iris['Species'])

    return model, iris.iloc[:, 0:4]


@pytest.mark.incremental
class TestModels:
    def test_register_astore(self, astore):
        from sasctl.tasks import register_model
        from sasctl import RestObj

        # Register model and ensure attributes are set correctly
        model = register_model(astore, ASTORE_MODEL_NAME,
                               project=PROJECT_NAME,
                               force=True)
        assert isinstance(model, RestObj)
        assert ASTORE_MODEL_NAME == model.name

    def test_register_sklearn(self, sklearn_model):
        from sasctl.tasks import register_model
        from sasctl import RestObj

        sk_model, train_df = sklearn_model

        # Register model and ensure attributes are set correctly
        model = register_model(sk_model, SCIKIT_MODEL_NAME,
                               project=PROJECT_NAME,
                               input=train_df,
                               force=True)
        assert isinstance(model, RestObj)
        assert SCIKIT_MODEL_NAME == model.name
        assert 'Classification' == model.function
        assert 'Logistic regression' == model.algorithm
        assert 'Python' == model.trainCodeType
        assert 'ds2MultiType' == model.scoreCodeType

        # Don't compare to sys.version since cassettes used may have been
        # created by a different version
        assert re.match('Python \d\.\d', model.tool)

        # Ensure input & output metadata was set
        for col in train_df.columns:
            assert 1 == len([v for v in model.inputVariables
                             + model.outputVariables if v.get('name') == col])

        # Ensure model files were created
        from sasctl.services import model_repository as mr
        files = mr.get_model_contents(model)
        filenames = [f.name for f in files]
        assert 'model.pkl' in filenames
        assert 'dmcas_espscorecode.sas' in filenames
        assert 'dmcas_packagescorecode.sas' in filenames

    def test_publish_sklearn(self):
        from sasctl.tasks import publish_model
        from sasctl.services import model_repository as mr

        model = mr.get_model(SCIKIT_MODEL_NAME, PROJECT_NAME)
        p = publish_model(model, 'maslocal', max_retries=100)

        # Score step should have been defined in the module
        assert 'score' in p.stepIds

        # MAS module should automatically have methods bound
        assert callable(p.score)

    def test_score_sklearn(self):
        from sasctl.services import microanalytic_score as mas

        m = mas.get_module(SCIKIT_MODEL_NAME.replace(' ', ''))
        m = mas.define_steps(m)
        r = m.score(sepalwidth=1, sepallength=2, petallength=3, petalwidth=4)
        assert r == 'virginica'

