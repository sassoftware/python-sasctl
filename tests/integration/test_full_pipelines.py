#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test multiple operations that must complete in order."""

import pytest

# Every test function in the module will automatically receive the session fixture
pytestmark = pytest.mark.usefixtures('session')


@pytest.mark.incremental
class TestAStoreRegressionModel:
    PROJECT_NAME = 'sasctl_testing Pipeline Project'
    MODEL_NAME = 'sasctl_testing Pipeline ASTORE Model'
    CAS_DEST = 'sasctl_test_cas'
    MAS_DEST = 'sasctl_test_mas'
    CAS_MODEL_TABLE = 'sasctl_test_model_table'

    def test_register_model(self, cas_session, boston_dataset):
        from sasctl import register_model

        TARGET = 'Price'

        # Upload the data to CAS
        tbl = cas_session.upload(boston_dataset).casTable

        # Create the model
        cas_session.loadactionset('regression')
        features = tbl.columns[tbl.columns != TARGET]
        tbl.glm(target=TARGET, inputs=list(features), savestate='model_table')
        astore = cas_session.CASTable('model_table')

        model = register_model(astore, self.MODEL_NAME, self.PROJECT_NAME, force=True)
        assert model.name == self.MODEL_NAME
        assert model.projectName == self.PROJECT_NAME
        assert model.function.lower() == 'prediction'
        assert model.algorithm.lower() == 'linear regression'
        assert model.targetVariable.lower() == 'price'

    def test_create_cas_destination(self):
        from sasctl.services import model_publish as mp

        dest = mp.create_cas_destination(self.CAS_DEST, 'Public', self.CAS_MODEL_TABLE)

        assert dest.name == self.CAS_DEST
        assert dest.destinationType == 'cas'
        assert dest.destinationTable == self.CAS_MODEL_TABLE

    def test_publish_cas(self, request):
        from sasctl import publish_model

        module = publish_model(self.MODEL_NAME, self.CAS_DEST)

        # Store module name so we can retrieve it in later tests
        request.config.cache.set('CAS_MODULE_NAME', module.name)

        assert module.state == 'completed'
        assert module.destinationName == self.CAS_DEST
        assert module.publishType == 'casModel'

    def test_score_cas(self, cas_session, boston_dataset, request):
        tbl = cas_session.upload(boston_dataset).casTable
        cas_session.loadactionset('modelpublishing')

        module_name = request.config.cache.get('CAS_MODULE_NAME', None)
        assert  module_name is not None

        result = cas_session.runModelLocal(modelName=module_name,
                                           modelTable=dict(name=self.CAS_MODEL_TABLE,
                                                           caslib='Public'),
                                           inTable=tbl,
                                           outTable=dict(name='boston_scored',
                                                         caslib='Public'))

        assert result.status_code == 0

        result = cas_session.CASTable('boston_scored', caslib='Public').head()

        assert 'P_Price' in result.columns

    def test_create_mas_destination(self):
        from sasctl.services import model_publish as mp

        dest = mp.create_mas_destination(self.MAS_DEST, 'localhost')

        assert dest.name == self.MAS_DEST
        assert dest.destinationType == 'microAnalyticService'
        assert dest.masUri == 'localhost'

    def test_publish_mas(self, request):
        from sasctl import publish_model

        module = publish_model(self.MODEL_NAME, 'maslocal')
        # module = publish_model(self.MODEL_NAME, self.MAS_DEST)

        # Store module name so we can retrieve it in later tests
        request.config.cache.set('MAS_MODULE_NAME', module.name)

        assert module.scope.lower() == 'public'
        assert hasattr(module, 'score')

    def test_score_mas(self, boston_dataset, request):
        from sasctl.services import microanalytic_score as mas

        module_name = request.config.cache.get('MAS_MODULE_NAME', None)
        assert module_name is not None

        # Retrieve the module from MAS
        module = mas.get_module(module_name)
        assert module.name == module_name

        # Create Python methods for the model steps
        module = mas.define_steps(module)

        assert hasattr(module, 'score')
        result = module.score(boston_dataset.iloc[0, :])

        assert round(result, 4) == 30.0038


@pytest.mark.incremental
class TestSklearnRegressionModel:
    PROJECT_NAME = 'sasctl_testing Pipeline Project'
    MODEL_NAME = 'sasctl_testing Pipeline Sklearn Model'
    CAS_DEST = 'sasctl_test_cas'
    MAS_DEST = 'sasctl_test_mas'
    CAS_MODEL_TABLE = 'sasctl_test_model_table'

    def test_register_model(self, boston_dataset):
        pytest.importorskip('sklearn')
        from sasctl import register_model
        from sklearn.ensemble import GradientBoostingRegressor

        TARGET = 'Price'

        X = boston_dataset.drop(TARGET, axis=1)
        y = boston_dataset[TARGET]

        model = GradientBoostingRegressor()
        model.fit(X, y)

        model = register_model(model, self.MODEL_NAME, self.PROJECT_NAME, input=X, force=True)
        assert model.name == self.MODEL_NAME
        assert model.projectName == self.PROJECT_NAME
        assert model.function.lower() == 'prediction'
        assert model.algorithm.lower() == 'gradient boosting'
        assert model.targetLevel.lower() == 'interval'
        assert model.tool.lower().startswith('python')

    def test_create_cas_destination(self):
        from sasctl.services import model_publish as mp

        dest = mp.get_destination(self.CAS_DEST)
        if not dest:
            dest = mp.create_cas_destination(self.CAS_DEST,
                                             'Public',
                                             self.CAS_MODEL_TABLE)

        assert dest.name == self.CAS_DEST
        assert dest.destinationType == 'cas'
        assert dest.destinationTable == self.CAS_MODEL_TABLE

    def test_publish_cas(self, request):
        from sasctl import publish_model

        module = publish_model(self.MODEL_NAME, self.CAS_DEST, replace=True)

        # Store module name so we can retrieve it in later tests
        request.config.cache.set('CAS_MODULE_NAME', module.name)

        assert module.state == 'completed'
        assert module.destinationName == self.CAS_DEST
        assert module.publishType == 'casModel'

    def test_score_cas(self, cas_session, boston_dataset, request):
        tbl = cas_session.upload(boston_dataset).casTable
        cas_session.loadactionset('modelpublishing')

        module_name = request.config.cache.get('CAS_MODULE_NAME', None)
        assert  module_name is not None

        result = cas_session.runModelLocal(modelName=module_name,
                                           modelTable=dict(name=self.CAS_MODEL_TABLE,
                                                           caslib='Public'),
                                           inTable=tbl,
                                           outTable=dict(name='boston_scored',
                                                         caslib='Public'))

        assert result.status_code == 0

        result = cas_session.CASTable('boston_scored', caslib='Public').head()

        assert 'var1' in result.columns
        assert result.shape[0] > 0
        assert all(result['var1'] > 15)

    def test_publish_mas(self, request):
        from sasctl import publish_model

        module = publish_model(self.MODEL_NAME, 'maslocal')
        # module = publish_model(self.MODEL_NAME, self.MAS_DEST)

        # Store module name so we can retrieve it in later tests
        request.config.cache.set('MAS_MODULE_NAME', module.name)

        assert module.scope.lower() == 'public'
        assert hasattr(module, 'predict')

    def test_score_mas(self, boston_dataset, request):
        from sasctl.services import microanalytic_score as mas

        module_name = request.config.cache.get('MAS_MODULE_NAME', None)
        assert module_name is not None

        # Retrieve the module from MAS
        module = mas.get_module(module_name)
        assert module.name == module_name

        # Create Python methods for the model steps
        module = mas.define_steps(module)

        assert hasattr(module, 'predict')
        result = module.predict(boston_dataset.iloc[0, :])

        # Don't think order of rows is guaranteed.
        assert isinstance(result, float)
        assert result > 1


@pytest.mark.incremental
class TestSklearnClassificationModel:
    PROJECT_NAME = 'sasctl_testing Pipeline Classification Project'
    MODEL_NAME = 'sasctl_testing Pipeline Sklearn Iris Model'
    CAS_DEST = 'sasctl_test_cas'
    MAS_DEST = 'sasctl_test_mas'
    CAS_MODEL_TABLE = 'sasctl_test_model_table'

    def test_register_model(self, iris_dataset):
        pytest.importorskip('sklearn')
        from sasctl import register_model
        from sklearn.ensemble import GradientBoostingClassifier

        TARGET = 'Species'

        X = iris_dataset.drop(TARGET, axis=1)
        y = iris_dataset[TARGET]

        model = GradientBoostingClassifier()
        model.fit(X, y)

        model = register_model(model, self.MODEL_NAME, self.PROJECT_NAME, input=X, force=True)
        assert model.name == self.MODEL_NAME
        assert model.projectName == self.PROJECT_NAME
        assert model.function.lower() == 'classification'
        assert model.algorithm.lower() == 'gradient boosting'
        assert model.tool.lower().startswith('python')

    def test_create_cas_destination(self):
        from sasctl.services import model_publish as mp

        dest = mp.get_destination(self.CAS_DEST)
        if not dest:
            dest = mp.create_cas_destination(self.CAS_DEST,
                                             'Public',
                                             self.CAS_MODEL_TABLE)

        assert dest.name == self.CAS_DEST
        assert dest.destinationType == 'cas'
        assert dest.destinationTable == self.CAS_MODEL_TABLE

    def test_publish_cas(self, request):
        from sasctl import publish_model

        module = publish_model(self.MODEL_NAME, self.CAS_DEST, replace=True)

        # Store module name so we can retrieve it in later tests
        request.config.cache.set('CAS_MODULE_NAME', module.name)

        assert module.state == 'completed'
        assert module.destinationName == self.CAS_DEST
        assert module.publishType == 'casModel'

    def test_publish_mas(self, request):
        from sasctl import publish_model

        module = publish_model(self.MODEL_NAME, 'maslocal', replace=True)

        # Store module name so we can retrieve it in later tests
        request.config.cache.set('MAS_MODULE_NAME', module.name)

        assert module.scope.lower() == 'public'
        assert hasattr(module, 'predict')

    def test_score_mas(self, iris_dataset, request):
        from sasctl.services import microanalytic_score as mas

        module_name = request.config.cache.get('MAS_MODULE_NAME', None)
        assert module_name is not None

        # Retrieve the module from MAS
        module = mas.get_module(module_name)
        assert module.name == module_name

        # Create Python methods for the model steps
        module = mas.define_steps(module)

        # Call .predict()
        x = iris_dataset.iloc[0, :]
        assert hasattr(module, 'predict')
        result = module.predict(x)
        assert isinstance(result, str)
        assert result in ('setosa', 'virginica', 'versicolor')

        # Call .predict_proba()
        assert hasattr(module, 'predict_proba')
        probs = module.predict_proba(x)
        assert len(probs) == 3
        assert all(isinstance(p, float) for p in probs)
        assert round(sum(probs), 5) == 1.0
