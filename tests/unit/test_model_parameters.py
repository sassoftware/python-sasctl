import copy
import json
from unittest import mock

import pytest
import warnings
import pandas as pd
import tempfile
from requests.models import Response
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

from sasctl.pzmm import ModelParameters as mp
from sasctl import current_session
from sasctl import RestObj

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
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        model.fit(X, y)
    return model


@pytest.mark.incremental
class TestSKLearnModel:
    PROJECT_NAME = "PZMM SKLearn Test Project"
    MODEL_NAME = "SKLearnModel"

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

    def test_update_kpis(self, sklearn_model):
        PROJECT = RestObj({"name": "Test Project", "id": "98765"})
        MODELS = [
            RestObj({"name": "TestModel1", "id": "12345", "projectId": PROJECT["id"]}),
            RestObj({"name": "TestModel2", "id": "67890", "projectId": PROJECT["id"]}),
        ]
        USER = "username"
        KPIS = pd.DataFrame({'ModelUUID': ["12345"], 'TestKPI': [1]})
        TESTFILE = {"hyperparameters": {"TEST": "1"}, "kpis": {}}
        FILE_RESPONSE = Response()
        FILE_RESPONSE.status_code = 200
        FILE_RESPONSE.body = json.dumps(TESTFILE).encode('utf-8')

        with mock.patch("sasctl.core.Session._get_authorization_token"):
            current_session("example.com", USER, "password")

        with mock.patch(
                "sasctl.pzmm.modelParameters.get_project_kpis"
        ) as get_project_kpis:
            with mock.patch(
                    "sasctl.pzmm.modelParameters._find_file"
            ) as _find_file:
                with mock.patch(
                        "sasctl._services.model_repository.add_model_content"
                ) as add_model_content:
                    get_project_kpis.return_value = copy.deepcopy(KPIS)
                    _find_file.return_value = copy.deepcopy(FILE_RESPONSE)
                    get_project_kpis.return_value = "Placeholder"

                    #basic update call
                    mp.update_kpis("98765")
                    assert add_model_content.call_args[2] == "TestModel1Hyperparameters.json"
                    