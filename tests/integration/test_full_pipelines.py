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
    CAS_MODULE_NAME = None # Will be set when module is created

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




#
# ####
# from sasctl.services import microanalytic_score as mas
# module2 = mas.get_module('LinearRegression')
# module2 = mas.define_steps(module2)
# result = module2.score(tbl.head(1))
# print('result')
# print('done')
# ###
#
#
# features = tbl.columns[tbl.columns != 'medv']
# tbl.glm(target='medv', inputs=list(features), savestate='model_table')
# astore = cas.CASTable('model_table')
#
# from sasctl.services import model_publish as mp
# mp.create_cas_destination('caslocal', 'Public', 'model_table')
#
# model = register_model(astore, 'Linear Regression', 'Boston Housing', force=True)
# module = publish_model(model, 'caslocal')
#
# cas.loadactionset('modelpublishing')
#
# cas.runModelLocal(modelName=module.name,
#      modelTable={"caslib":"Public", "name":"model_table"},
#      inTable=tbl,
#      outTable={"caslib":"Public", "name":"boston_scored"})
#
# cas.CASTable('hmeqscored', caslib='Public').head()
#
#
# module2 = publish_model(model, 'maslocal')
# result = module2.score(**tbl.head().iloc[0, :].to_dict())
# print(result)
#
#
# # Need to be able to specify input table (DATA STEP set statement)
# cas.loadactionset('modelpublishing')
# cas.modelPublishing.runModelLocal(modelTable="modeltable_cars", modelName="model1")
#
#
# # explicit host since SWAT converts to "sasserver.demo.sas.com"
# with Session(hostname, 'sasdemo', 'Orion123'):
#     model = register_model(astore, 'Linear Regression', 'Boston Housing', force=True)
#
#     module = publish_model(model, 'maslocal')
#     response = mas.execute_module_step(module, 'score',
#                                        SepalLength=5.1,
#                                        SepalWidth=3.5,
#                                        PetalLength=1.4,
#                                        PetalWidth=0.2)
#
# ######
#
#
# #Create open source pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
#
# #Note, I needed to add impute becuase data had missing values
# openmodel = Pipeline([("imputer", SimpleImputer(strategy="constant")),
#                       ("logreg", LogisticRegression()) ])
#
# #Fix model
# openmodel.fit(hmeq,y)
#
# #Register Linear Regression model
# # Register the model in SAS Model Manager
# model=register_model(openmodel,
#                model_name,
#                project=project,
#                input=hmeq,
#                force=True)          # Create project if it doesn't exist
#
#
# # Publish the model to the batch scoring engine (CAS)
# module_cas = publish_model(model_name, 'caslocal', replace=True)
#
# #Batch Score in CAS data using publish score.
# sas.loadactionset('modelPublishing')
# sas.runModelLocal(modelName=module_cas['name'],
#      modelTable={"caslib":"Public", "name":"SAS_MODEL_TABLE"},
#      inTable= {"caslib":"Public", "name":"hmeq"},
#      outTable={"caslib":"Public", "name":"hmeqscored"})
# sas.table.fetch(table={"name":"hmeqscored","caslib":"Public"},
#                maxRows=10)