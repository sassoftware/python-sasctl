#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pandas as pd
import pytest

from sasctl.core import RestObj
from sasctl._services.model_repository import ModelRepository


# invlid model
# parameterize with multiple algorithms?
# binary classification
# multiclass classification
# regression
# pipeline
# gridsearch
# check numpy inputs & outputs
# check model with sparse arrays

# correct calls to pzmm made and correct data passed.


def test_sklearn_metadata():
    """Verify that meta data is correctly extracted from various scikit-learn models."""
    pytest.importorskip("sklearn")

    from sasctl.tasks import _sklearn_to_dict
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    info = _sklearn_to_dict(LinearRegression())
    assert info["algorithm"] == "Linear regression"
    assert info["function"] == "prediction"

    info = _sklearn_to_dict(LogisticRegression())
    assert info["algorithm"] == "Logistic regression"
    assert info["function"] == "classification"

    info = _sklearn_to_dict(SVC())
    assert info["algorithm"] == "Support vector machine"
    assert info["function"] == "classification"

    info = _sklearn_to_dict(GradientBoostingClassifier())
    assert info["algorithm"] == "Gradient boosting"
    assert info["function"] == "classification"

    info = _sklearn_to_dict(DecisionTreeClassifier())
    assert info["algorithm"] == "Decision tree"
    assert info["function"] == "classification"

    info = _sklearn_to_dict(RandomForestClassifier())
    assert info["algorithm"] == "Forest"
    assert info["function"] == "classification"


def test_register_sklearn_with_pzmm(iris_dataset):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sasctl.tasks import _register_sklearn_40

    target = "Species"
    X = iris_dataset.drop(columns=target)
    y = iris_dataset[target]

    model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    model.fit(X, y)

    MODEL_NAME = "Iris Logistic"
    PROJECT_NAME = "Spam"

    # Mock pzmm import_model call so there's no dependency on a Viya server.
    with mock.patch("sasctl.pzmm.ImportModel.import_model") as import_model:
        _register_sklearn_40(model, MODEL_NAME, PROJECT_NAME, X, y)

    assert import_model.call_count == 1
    args, kwargs = import_model.call_args

    # Verify that expected files were generated.
    files = kwargs["model_files"]
    assert isinstance(files, dict)
    assert all(f in files for f in (f"{MODEL_NAME}.pickle", "inputVar.json", "outputVar.json", "ModelProperties.json", "fileMetadata.json"))

    assert kwargs["model_prefix"] == MODEL_NAME
    assert kwargs["project"] == PROJECT_NAME
    assert kwargs["predict_method"] == model.predict
    assert kwargs["output_variables"]
    assert kwargs["score_cas"] == True
    assert kwargs["missing_values"] == False

    pd.testing.assert_frame_equal(kwargs["input_data"], X)

    pytest.fail("Verify import_model inputs are correct")

"""
        metrics : string list
            The scoring metrics for the model. For classification models, it is assumed
            that the first value in the list represents the classification output. This
            function supports single- and multi-class classification models.
        project_version : string, optional
            The project version to import the model in to on SAS Model Manager. The
            default value is "latest".
        overwrite_model : bool, optional
            Set whether models with the same name should be overwritten when attempting
            to import the model. The default value is False.
        predict_threshold : float, optional
            The prediction threshold for normalized probability metrics. Values are
            expected to be between 0 and 1. The default value is None.
        target_values : list of strings, optional
            A list of target values for the target variable. This argument and the
            metrics argument dictate the handling of the predicted values from the
            prediction method. The default value is None.
        kwargs : dict, optional
            Other keyword arguments are passed to the following function:
            * sasctl.pzmm.ScoreCode.write_score_code(...,
                binary_h2o_model=False,
                binary_string=None,
                model_file_name=None,
                mojo_model=False,
                statsmodels_model=False
            )
"""


def test_parse_module_url():
    from sasctl.tasks import _parse_module_url

    body = RestObj(
        {
            "createdBy": "sasdemo",
            "creationTimeStamp": "2019-08-26T15:16:42.900Z",
            "destinationName": "maslocal",
            "id": "62cae262-7287-412b-8f1d-bd2a12c8b434",
            "links": [
                {
                    "href": "/modelPublish/models/44d526bc-d513-4637-b8a7-72daee4a7730",
                    "method": "GET",
                    "rel": "up",
                    "type": "application/vnd.sas.models.publishing.publish",
                    "uri": "/modelPublish/models/44d526bc-d513-4637-b8a7-72daee4a7730",
                },
                {
                    "href": "/modelPublish/models/44d526bc-d513-4637-b8a7-72daee4a7730/log",
                    "method": "GET",
                    "rel": "self",
                    "type": "application/json",
                    "uri": "/modelPublish/models/44d526bc-d513-4637-b8a7-72daee4a7730/log",
                },
            ],
            "log": 'SUCCESS==={"links":[{"method":"GET","rel":"up","href":"/microanalyticScore/jobs","uri":"/microanalyticScore/jobs","type":"application/vnd.sas.collection","itemType":"application/vnd.sas.microanalytic.job"},{"method":"GET","rel":"self","href":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3","uri":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3","type":"application/vnd.sas.microanalytic.job"},{"method":"GET","rel":"source","href":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3/source","uri":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3/source","type":"application/vnd.sas.microanalytic.module.source"},{"method":"GET","rel":"submodules","href":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3/submodules","uri":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3/submodules","type":"application/vnd.sas.collection","itemType":"application/vnd.sas.microanalytic.submodule"},{"method":"DELETE","rel":"delete","href":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3","uri":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3"},{"method":"GET","rel":"module","href":"/microanalyticScore/modules/decisiontree","uri":"/microanalyticScore/modules/decisiontree","type":"application/vnd.sas.microanalytic.module"}],"version":1,"createdBy":"sasdemo","creationTimeStamp":"2019-08-26T15:16:42.857Z","modifiedBy":"sasdemo","modifiedTimeStamp":"2019-08-26T15:16:48.988Z","id":"465ecad8-cfd0-4403-ac8a-e49cd248fae3","moduleId":"decisiontree","state":"completed","errors":[]}',
            "modelId": "459aae0d-d64f-4376-94e7-be31911f4bdb",
            "modelName": "DecisionTree",
            "modifiedBy": "sasdemo",
            "modifiedTimeStamp": "2019-08-26T15:16:49.315Z",
            "publishName": "Decision Tree",
            "version": 1,
        }
    )

    msg = body.get("log").lstrip("SUCßCESS===")
    assert _parse_module_url(msg) == "/microanalyticScore/modules/decisiontree"


def test_save_performance_project_types():
    from sasctl.tasks import update_model_performance

    with mock.patch(
        "sasctl._services.model_repository.ModelRepository.get_model"
    ) as model:
        with mock.patch(
            "sasctl._services.model_repository.ModelRepository.get_project"
        ) as project:
            model.return_value = RestObj(name="fakemodel", projectId=1)

            # Function is required
            with pytest.raises(ValueError):
                project.return_value = {}
                update_model_performance(None, None, None)

            # Target Level is required
            with pytest.raises(ValueError):
                project.return_value = {"function": "Prediction"}
                update_model_performance(None, None, None)

            # Prediction variable required
            with pytest.raises(ValueError):
                project.return_value = {
                    "function": "Prediction",
                    "targetLevel": "Binary",
                }
                update_model_performance(None, None, None)

            # Classification variable required
            with pytest.raises(ValueError):
                project.return_value = {
                    "function": "classification",
                    "targetLevel": "Binary",
                }
                update_model_performance(None, None, None)

    # Check projects w/ invalid properties


@mock.patch.object(ModelRepository, "list_repositories")
@mock.patch.object(ModelRepository, "get_project")
def test_register_model_403_error(get_project, list_repositories):
    """Verify HTTP 403 is converted to a user-friendly error.

    Depending on environment configuration, this can happen when attempting to
    find a repository.

    See: https://github.com/sassoftware/python-sasctl/issues/39
    """
    from urllib.error import HTTPError
    from sasctl.exceptions import AuthorizationError
    from sasctl.tasks import register_model

    get_project.return_value = {"name": "Project Name"}
    list_repositories.side_effect = HTTPError(None, 403, None, None, None)

    # HTTP 403 error when getting repository should throw a user-friendly
    # AuthorizationError
    with pytest.raises(AuthorizationError):
        register_model(None, "model name", "project name")

    # All other errors should be bubbled up
    list_repositories.side_effect = HTTPError(None, 404, None, None, None)
    with pytest.raises(HTTPError):
        register_model(None, "model name", "project name")
