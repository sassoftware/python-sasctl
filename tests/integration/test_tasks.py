#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import warnings
from unittest import mock

import pytest


# Every test function in the module will automatically receive the session fixture
pytestmark = pytest.mark.usefixtures("session")

ASTORE_MODEL_NAME = "sasctl_testing_astore_model"
SCIKIT_MODEL_NAME = "sasctl_testing_scikit_model"
PROJECT_NAME = "sasctl_testing_task_project"


@pytest.fixture
def sklearn_logistic_model():
    """A Scikit-Learn logistic regression fit to Iris data set."""

    pd = pytest.importorskip("pandas")

    try:
        from sklearn import datasets
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("Package `sklearn` not found.")

    raw = datasets.load_iris()
    iris = pd.DataFrame(raw.data, columns=raw.feature_names)
    iris = iris.join(pd.DataFrame(raw.target))
    iris.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
    iris["Species"] = iris["Species"].astype("category")
    iris.Species.cat.categories = raw.target_names

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        model.fit(iris.iloc[:, 0:4], iris["Species"])

    return model, iris.iloc[:, 0:4]


@pytest.fixture
def sklearn_linear_model(boston_dataset):
    """A Scikit-Learn linear regression fit to Boston housing data."""

    pd = pytest.importorskip("pandas")
    linear_model = pytest.importorskip("sklearn.linear_model")

    X = boston_dataset.drop(columns=["Price"])
    y = boston_dataset["Price"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm = linear_model.LinearRegression()
        lm.fit(X, y)

    return lm, X, y


@pytest.mark.incremental
class TestModels:
    def test_register_astore(self, iris_astore):
        from sasctl.tasks import register_model
        from sasctl import RestObj

        # Register model and ensure attributes are set correctly
        model = register_model(
            iris_astore, ASTORE_MODEL_NAME, project=PROJECT_NAME, force=True
        )
        assert isinstance(model, RestObj)
        assert ASTORE_MODEL_NAME == model.name

    def test_created_project(self):
        from sasctl.services import model_repository as mr

        project = mr.get_project(PROJECT_NAME)
        assert project.function.lower() == "classification"
        assert "Species" in project.eventProbabilityVariable

    def test_register_sklearn(self, sklearn_logistic_model):
        from sasctl.tasks import register_model
        from sasctl import RestObj

        sk_model, train_df = sklearn_logistic_model

        # Register model and ensure attributes are set correctly
        model = register_model(
            sk_model,
            SCIKIT_MODEL_NAME,
            project=PROJECT_NAME,
            input=train_df,
            force=True,
        )
        assert isinstance(model, RestObj)
        assert SCIKIT_MODEL_NAME == model.name
        assert "classification" == model.function.lower()
        assert "Logistic regression" == model.algorithm
        assert "Python" == model.trainCodeType
        assert "ds2MultiType" == model.scoreCodeType
        assert len(model.inputVariables) == 4
        assert len(model.outputVariables) == 1

        # Don't compare to sys.version since cassettes used may have been
        # created by a different version
        assert re.match(r"Python \d\.\d", model.tool)

        # Ensure input & output metadata was set
        for col in train_df.columns:
            assert 1 == len(
                [
                    v
                    for v in model.inputVariables + model.outputVariables
                    if v.get("name") == col
                ]
            )

        # Ensure model files were created
        from sasctl.services import model_repository as mr

        files = mr.get_model_contents(model)
        filenames = [f.name for f in files]
        assert "model.pkl" in filenames
        assert "dmcas_epscorecode.sas" in filenames
        assert "dmcas_packagescorecode.sas" in filenames

    def test_publish_sklearn(self):
        from sasctl.tasks import publish_model
        from sasctl.services import model_repository as mr

        model = mr.get_model(SCIKIT_MODEL_NAME, PROJECT_NAME)
        p = publish_model(model, "maslocal", max_retries=100)

        # Model functions should have been defined in the module
        assert "predict" in p.stepIds
        assert "predict_proba" in p.stepIds

        # MAS module should automatically have methods bound
        assert callable(p.predict)
        assert callable(p.predict_proba)

    def test_publish_sklearn_again(self, cache):
        from sasctl.tasks import publish_model
        from sasctl.services import model_repository as mr

        model = mr.get_model(SCIKIT_MODEL_NAME, PROJECT_NAME)

        # Should not be able to republish the model by default
        with pytest.raises(RuntimeError):
            publish_model(model, "maslocal", max_retries=100)

        # Publish should succeed with replace flag
        p = publish_model(model, "maslocal", max_retries=100, replace=True)

        # Module name in MAS may not exactly match name of model.  Store the assigned name of the model as
        # it appears in MAS so we can call it in subsequent test steps.
        cache.set("MAS_MODULE_NAME", p.name)

        # Model functions should have been defined in the module
        assert "predict" in p.stepIds
        assert "predict_proba" in p.stepIds

        # MAS module should automatically have methods bound
        assert callable(p.predict)
        assert callable(p.predict_proba)

    def test_score_sklearn(self, cache):
        from sasctl.services import microanalytic_score as mas

        # Read the expected MAS module name from the cache.  Should have been placed there after publishing
        # in the previous test step.
        module_name = cache.get("MAS_MODULE_NAME", None)

        m = mas.get_module(module_name)
        m = mas.define_steps(m)
        r = m.predict(sepalwidth=1, sepallength=2, petallength=3, petalwidth=4)
        assert r == "virginica"


@pytest.mark.incremental
class TestSklearnLinearModel:
    MODEL_NAME = "sasctl_testing Scikit Linear Model"
    PROJECT_NAME = "sasctl_testing Boston Housing"

    def test_register_model(self, sklearn_linear_model):
        from sasctl.tasks import register_model
        from sasctl import RestObj

        sk_model, X, _ = sklearn_linear_model

        # Register model and ensure attributes are set correctly
        model = register_model(
            sk_model, self.MODEL_NAME, project=self.PROJECT_NAME, input=X, force=True
        )

        assert isinstance(model, RestObj)
        assert self.MODEL_NAME == model.name
        assert "prediction" == model.function.lower()
        assert "Linear regression" == model.algorithm
        assert "Python" == model.trainCodeType
        assert "ds2MultiType" == model.scoreCodeType

        assert len(model.inputVariables) == 13
        assert len(model.outputVariables) == 1

        # Don't compare to sys.version since cassettes used may have been
        # created by a different version
        assert re.match(r"Python \d\.\d", model.tool)

        # Ensure input & output metadata was set
        for col in X.columns:
            assert 1 == len(
                [
                    v
                    for v in model.inputVariables + model.outputVariables
                    if v.get("name") == col
                ]
            )

        # Ensure model files were created
        from sasctl.services import model_repository as mr

        files = mr.get_model_contents(model)
        filenames = [f.name for f in files]
        assert "model.pkl" in filenames
        assert "dmcas_epscorecode.sas" in filenames
        assert "dmcas_packagescorecode.sas" in filenames

    def test_create_performance_definition(self):
        from sasctl.services import model_repository as mr
        from sasctl.services import model_management as mm

        project = mr.get_project(self.PROJECT_NAME)
        # Update project properties
        project["function"] = "prediction"
        project["targetLevel"] = "interval"
        project["targetVariable"] = "Price"
        project["predictionVariable"] = "var1"
        project = mr.update_project(project)

        mm.create_performance_definition(
            models=[self.MODEL_NAME], table_prefix="boston", library_name="Public"
        )

    def test_update_model_performance(self, sklearn_linear_model, cas_session):
        from sasctl.tasks import update_model_performance

        lm, X, y = sklearn_linear_model

        # Score & set output var
        train_df = X.copy()
        train_df["var1"] = lm.predict(X)
        train_df["Price"] = y

        with mock.patch("swat.CAS") as CAS:
            CAS.return_value = cas_session

            # NOTE: can only automate testing of 1 period at a time since
            # upload_model_performance closes the CAS session when it's done.
            for period in ["q12019"]:
                sample = train_df.sample(frac=0.1)
                update_model_performance(sample, self.MODEL_NAME, period)
