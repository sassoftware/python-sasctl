import ast
import copy
import json
import tempfile
import warnings
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from requests.models import Response
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from sasctl import RestObj, current_session
from sasctl.pzmm import ModelParameters as mp
import unittest
import uuid
import numpy as np


class BadModel:
    attr = None


@pytest.fixture
def bad_model():
    return BadModel()


@pytest.fixture
def train_data():
    """Returns the Iris data set as (X, y)"""
    raw = datasets.load_iris()
    iris = pd.DataFrame(raw.data, columns=raw.feature_names)
    iris = iris.join(pd.DataFrame(raw.target))
    iris.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
    iris["Species"] = iris["Species"].astype("category")
    iris.Species.cat.categories = raw.target_names
    return iris.iloc[:, 0:4], iris["Species"]


@pytest.fixture
def sklearn_model(train_data):
    """Returns a simple Scikit-Learn model"""
    X, y = train_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000
        )
        model.fit(X, y)
    return model


class TestSKLearnModel:
    USER = "username"
    PROJECT_NAME = "PZMM SKLearn Test Project"
    MODEL_NAME = "SKLearnModel"
    PROJECT = RestObj({"name": "Test Project", "id": "98765"})
    MODELS = [
        RestObj({"name": "TestModel1", "id": "12345", "projectId": PROJECT["id"]}),
        RestObj({"name": "TestModel2", "id": "67890", "projectId": PROJECT["id"]}),
    ]
    KPIS = pd.DataFrame(
        {"ModelUUID": ["12345", "00000"], "TestKPI": [1, 9], "TimeLabel": [0, 1]}
    )
    TESTJSON = {"hyperparameters": {"TEST": "1"}}
    MODEL_FILES = [RestObj({"name": "file1"}), RestObj({"name": "file2"})]
    COLS = [
        "TimeSK",
        "TimeLabel",
        "ProjectUUID",
        "ModelName",
        "ModelUUID",
        "ModelFlag",
        "_AUC_",
        "_F1_",
        "_TPR_",
        "_FPR_",
    ]

    def test_generate_hyperparameters(self, sklearn_model):
        tmp_dir = tempfile.TemporaryDirectory()
        mp.generate_hyperparameters(sklearn_model, self.MODEL_NAME, Path(tmp_dir.name))
        assert Path(
            Path(tmp_dir.name) / f"./{self.MODEL_NAME}Hyperparameters.json"
        ).exists()

    def test_bad_model_hyperparameters(self, bad_model):
        tmp_dir = tempfile.TemporaryDirectory()
        with pytest.raises(ValueError):
            mp.generate_hyperparameters(bad_model, self.MODEL_NAME, Path(tmp_dir.name))

    def test_update_json(self):
        from sasctl.pzmm.model_parameters import ModelParameters as mp

        # ensure that only relevant rows are added to hyperparameter json

        input_json = copy.deepcopy(self.TESTJSON)
        input_kpis = copy.deepcopy(self.KPIS)
        assert (
            mp._update_json(self.MODELS[1]["id"], input_json, input_kpis)
            == self.TESTJSON
        )

        input_json = copy.deepcopy(self.TESTJSON)
        input_kpis = copy.deepcopy(self.KPIS)
        updated_json = mp._update_json(self.MODELS[0]["id"], input_json, input_kpis)

        pd.testing.assert_frame_equal(input_kpis, self.KPIS)
        assert "hyperparameters" in updated_json
        assert updated_json["hyperparameters"] == self.TESTJSON["hyperparameters"]
        assert "kpis" in updated_json
        assert len(updated_json["kpis"]) == 1
        assert updated_json["kpis"] == {"0": {"TestKPI": 1}}
        assert "TimeLabel" not in updated_json["kpis"]

    def test_find_file(self):
        FILE_RESPONSE = Response()
        FILE_RESPONSE.status_code = 200
        FILE_RESPONSE.body = json.dumps(self.TESTJSON).encode("utf-8")

        from sasctl.pzmm.model_parameters import _find_file

        with mock.patch("sasctl.core.Session._get_authorization_token"):
            current_session("example.com", self.USER, "password")

        with mock.patch(
            "sasctl._services.model_repository.ModelRepository.get_model_contents"
        ) as get_model_contents:
            get_model_contents.return_value = copy.deepcopy(self.MODEL_FILES)
            with pytest.raises(ValueError):
                _find_file(self.MODEL_NAME, "file0")

    def test_add_hyperparamters(self):
        with mock.patch("sasctl.core.Session._get_authorization_token"):
            current_session("example.com", self.USER, "password")

        with mock.patch(
            "sasctl._services.model_repository.ModelRepository" ".add_model_content"
        ) as add_model_content:
            with mock.patch(
                "sasctl.pzmm.model_parameters.ModelParameters.get_hyperparameters"
            ) as get_hyperparameters:
                with mock.patch(
                    "sasctl._services.model_repository.ModelRepository.get_model"
                ) as get_model:
                    # TODO: test if model that doesn't exist gets passed into the system
                    get_model.return_value = copy.deepcopy(self.MODELS[0])
                    get_hyperparameters.return_value = (
                        copy.deepcopy(self.TESTJSON),
                        "TestModel1Hyperparameters.json",
                    )

                    # ensure that argument with no kwargs returns same file
                    mp.add_hyperparameters("TestModel1")
                    test = add_model_content.call_args_list[0]
                    assert (
                        json.loads(add_model_content.call_args_list[0][0][1])
                        == self.TESTJSON
                    )

                    # ensure that kwarg is added properly
                    mp.add_hyperparameters("TestModel1", test_parameter="1")
                    assert (
                        json.loads(add_model_content.call_args_list[1][0][1])[
                            "hyperparameters"
                        ]["test_parameter"]
                        == "1"
                    )

                    # ensure that overwriting works properly

                    mp.add_hyperparameters("TestModel1", TEST="2")
                    assert (
                        json.loads(add_model_content.call_args_list[2][0][1])[
                            "hyperparameters"
                        ]["TEST"]
                        == "2"
                    )

    def test_get_project_kpis(self):
        cols = Response()
        cols.status_code = 200
        cols._content = json.dumps({"items": [{"name": "testName"}]}).encode("utf-8")

        rows = Response()
        rows.status_code = 200
        rows._content = json.dumps({"items": {"cells": ["testValue"]}}).encode("utf-8")
        with mock.patch("sasctl.core.Session._get_authorization_token"):
            current_session("example.com", self.USER, "password")

        with mock.patch(
            "sasctl._services.model_repository.ModelRepository" ".get_project"
        ) as get_project:
            with mock.patch("sasctl.core.Session.get") as get:
                get.side_effect = [cols, rows]
                get_project.return_value = self.PROJECT
                kpi_table = mp.get_project_kpis("Project")
                # ensure basic table is made correctly
                assert kpi_table.columns[0] == "testName"
                assert kpi_table.loc[0]["testName"] == "testValue"
                assert len(kpi_table) == 1
                assert len(kpi_table.columns) == 1

                get.reset_mock(side_effect=True)
                get.side_effect = [cols, rows]
                rows._content = json.dumps(
                    {"items": [{"cells": "testValue"}, {"cells": "."}]}
                ).encode("utf-8")

                kpi_table = mp.get_project_kpis("Project")

                assert len(kpi_table) == 2
                assert len(kpi_table.columns) == 1

                assert not kpi_table.loc[1]["testName"]


class TestSyncModelProperties(unittest.TestCase):
    MODEL_PROPERTIES = [
        ("targetVariable", "targetVariable"),
        ("targetLevel", "targetLevel"),
        ("targetEventValue", "targetEvent"),
        ("eventProbabilityVariable", "eventProbVar"),
        ("function", "function"),
    ]

    def test_project_id(self):
        with mock.patch(
            "sasctl._services.model_repository.ModelRepository.get_project"
        ) as get_project:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.get"
            ) as get:
                with mock.patch(
                    "sasctl._services.model_repository.ModelRepository.get_model"
                ) as get_model:
                    with mock.patch(
                        "sasctl._services.model_repository.ModelRepository.update_model"
                    ) as update:
                        pUUID = uuid.uuid4()
                        mp.sync_model_properties(pUUID, False)
                        get.assert_called_with(f"/projects/{pUUID}/models")

                        project_dict = {"id": "projectID"}
                        mp.sync_model_properties(project_dict, False)
                        get.assert_called_with("/projects/projectID/models")

                        project_name = "project"
                        get_project.return_value = {"id": "pid"}
                        mp.sync_model_properties(project_name, False)
                        get.assert_called_with("/projects/pid/models")

    def test_overwrite(self):
        with mock.patch(
            "sasctl._services.model_repository.ModelRepository.get_project"
        ) as get_project:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.get"
            ) as get:
                with mock.patch(
                    "sasctl._services.model_repository.ModelRepository.get_model"
                ) as get_model:
                    with mock.patch(
                        "sasctl._services.model_repository.ModelRepository.update_model"
                    ) as update:
                        project_dict = {
                            "id": "projectID",
                            "function": "project_function",
                            "targetLevel": "1",
                        }
                        get.return_value = [RestObj({"id": "modelID"})]
                        get_model.return_value = {"function": "classification"}
                        mp.sync_model_properties(project_dict)
                        update.assert_called_with(
                            {"function": "classification", "targetLevel": "1"}
                        )

                        project_dict = {
                            "id": "projectID",
                            "function": "project_function",
                            "targetLevel": "1",
                        }
                        get.return_value = [RestObj({"id": "modelID"})]
                        get_model.return_value = {"function": "classification"}
                        mp.sync_model_properties(project_dict, True)
                        update.assert_called_with(
                            {"function": "project_function", "targetLevel": "1"}
                        )


class TestGenerateHyperparameters(unittest.TestCase):
    def test_xgboost(self):
        xgboost = pytest.importorskip("xgboost")
        model = unittest.mock.Mock()
        model.__class__ = xgboost.Booster
        attrs = {"save_config.return_value": json.dumps({"test": "passed"})}
        model.configure_mock(**attrs)
        tmp_dir = tempfile.TemporaryDirectory()
        mp.generate_hyperparameters(model, "prefix", Path(tmp_dir.name))
        assert Path(Path(tmp_dir.name) / f"./prefixHyperparameters.json").exists()

    def test_xgboost_sklearn(self):
        xgboost = pytest.importorskip("xgboost")
        model = unittest.mock.Mock()
        model.__class__ = xgboost.XGBModel
        attrs = {"get_params.return_value": json.dumps({"test": "passed"})}
        model.configure_mock(**attrs)
        tmp_dir = tempfile.TemporaryDirectory()
        mp.generate_hyperparameters(model, "prefix", Path(tmp_dir.name))
        assert Path(Path(tmp_dir.name) / f"./prefixHyperparameters.json").exists()

    def test_h2o(self):
        h2o = pytest.importorskip("h2o")
        model = unittest.mock.Mock()
        model.__class__ = h2o.H2OFrame
        attrs = {"get_params.return_value": json.dumps({"test": "passed"})}
        model.configure_mock(**attrs)
        tmp_dir = tempfile.TemporaryDirectory()
        mp.generate_hyperparameters(model, "prefix", Path(tmp_dir.name))
        assert Path(Path(tmp_dir.name) / f"./prefixHyperparameters.json").exists()

    def test_tensorflow(self):
        tf = pytest.importorskip("tensorflow")
        model = unittest.mock.Mock()
        model.__class__ = tf.keras.Sequential
        attrs = {"get_config.return_value": json.dumps({"test": "passed"})}
        model.configure_mock(**attrs)
        tmp_dir = tempfile.TemporaryDirectory()
        mp.generate_hyperparameters(model, "prefix", Path(tmp_dir.name))
        assert Path(Path(tmp_dir.name) / f"./prefixHyperparameters.json").exists()

    def test_statsmodels(self):
        smf = pytest.importorskip("statsmodels.formula.api")
        model = unittest.mock.Mock(
            exog_names=["test", "exog"], weights=np.array([0, 1])
        )
        model.__class__ = smf.ols
        tmp_dir = tempfile.TemporaryDirectory()
        mp.generate_hyperparameters(model, "prefix", Path(tmp_dir.name))
        assert Path(Path(tmp_dir.name) / f"./prefixHyperparameters.json").exists()
