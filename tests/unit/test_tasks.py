#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock, TestCase


import pytest

from sasctl._services.model_repository import ModelRepository
from sasctl.core import RestObj
from sasctl.tasks import (
    _format_properties,
    _update_properties,
    _create_project,
    _compare_properties,
)


def test_sklearn_metadata():
    pytest.importorskip("sklearn")

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    from sasctl.tasks import _sklearn_to_dict

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


class TestFormatProperties(TestCase):
    _VARIABLE_PROPERTIES = ["name", "role", "type", "level", "length"]
    MODEL_PROPERTIES = [
        "function",
        "targetLevel",
        "targetVariable",
        "classTargetValues",
    ]

    def setUp(self):
        self.model = {
            "function": "Classification",
            "targetLevel": "Binary",
            "targetVariable": "BAD",
            "targetEvent": "1",
            "classTargetValues": "1, 0",
            "algorithm": "logistic",
        }

        self.input_var = [
            {
                "name": "input_test",
                "role": "the_dice",
                "type": "writer",
                "level": "high",
                "length": "long",
            }
        ]

        self.output_var = [
            {
                "name": "output_test",
                "role": "in_the_deep",
                "type": "mine",
                "level": "low",
                "length": "short",
            }
        ]

    def test_format_properties(self):
        properties, _ = _format_properties(self.model)
        for p in self.MODEL_PROPERTIES:
            self.assertEqual(properties[p], self.model[p])
        self.assertEqual(properties["targetEventValue"], self.model["targetEvent"])

    def test_format_properties_extras(self):
        properties, _ = _format_properties(self.model)
        self.assertEqual(properties.get("algorithm"), None)

    def test_format_properties_target_level(self):
        del self.model["targetLevel"]
        properties, _ = _format_properties(self.model)
        self.assertEqual(properties["targetLevel"], "Binary")
        self.assertEqual(self.model.get("targetLevel"), None)

        self.model["function"] = "predIction"
        self.model["algorithm"] = "a_regression_algo"
        properties, _ = _format_properties(self.model)
        self.assertEqual(properties["targetLevel"], "Interval")

        self.model["function"] = "something_else"
        properties, _ = _format_properties(self.model)
        self.assertEqual(properties["targetLevel"], "")

    def test_format_properties_input_variables(self):
        _, variables = _format_properties(self.model, self.input_var)
        self.assertEqual(len(variables), 1)

        for p in self._VARIABLE_PROPERTIES:
            self.assertEqual(variables[0][p], self.input_var[0][p])

    def test_format_properties_output_variables(self):
        _, variables = _format_properties(self.model, output_vars=self.output_var)
        self.assertEqual(len(variables), 1)

        for p in self._VARIABLE_PROPERTIES:
            self.assertEqual(variables[0][p], self.output_var[0][p])

    def test_format_properties_both_variables(self):
        _, variables = _format_properties(self.model, self.input_var, self.output_var)
        self.assertEqual(len(variables), 2)

        for p in self._VARIABLE_PROPERTIES:
            self.assertEqual(variables[0][p], self.input_var[0][p])
            self.assertEqual(variables[1][p], self.output_var[0][p])

    def test_format_properties_set_prediction_var(self):
        properties, _ = _format_properties(self.model, self.input_var, self.output_var)
        self.assertEqual(properties.get("eventProbabilityVariable"), "output_test")

        self.model["function"] = "prediction"
        properties, _ = _format_properties(self.model, self.input_var, self.output_var)
        self.assertEqual(properties.get("predictionVariable"), "output_test")


class TestCompareProperties(TestCase):
    properties = {
        "function": "Classification",
        "targetLevel": "Binary",
        "targetVariable": "BAD",
        "targetEvent": "1",
        "classTargetValues": "1, 0",
    }

    different_case_properties = {
        "function": "classification",
        "targetLevel": "binary",
        "targetVariable": "bad",
        "targetEvent": "1",
        "classTargetValues": "1, 0",
    }

    different_properties = {
        "function": "not_classification",
        "targetLevel": "not_binary",
        "targetVariable": "good",
        "targetEvent": "0",
        "classTargetValues": "0, 1",
    }

    def test_compare_properties_same(self):
        with mock.patch("sasctl.tasks._format_properties") as format_properties:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.get_project"
            ) as project:
                project.return_value = RestObj(**self.properties)
                format_properties.return_value = (self.properties, None)
                _compare_properties([], [])

    def test_compare_different_properties(self):
        with mock.patch("sasctl.tasks._format_properties") as format_properties:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.get_project"
            ) as project:
                project.return_value = RestObj(**self.properties)
                format_properties.return_value = (self.different_properties, None)
                self.assertWarns(
                    Warning, _compare_properties, project_name=[], model=[]
                )

    def test_compare_no_properties(self):
        with mock.patch("sasctl.tasks._format_properties") as format_properties:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.get_project"
            ) as project:
                project.return_value = RestObj()
                format_properties.return_value = (self.different_properties, None)
                self.assertWarns(
                    Warning, _compare_properties, project_name=[], model=[]
                )


class TestCreateProject(TestCase):
    def test_create_project_no_update(self):
        with mock.patch("sasctl.tasks._format_properties") as format_properties:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.create_project"
            ) as project:
                with mock.patch(
                    "sasctl._services.model_repository.ModelRepository.update_project"
                ) as update:
                    format_properties.return_value = ({"property": "test"}, "variable")
                    project.return_value = RestObj({"new": "project"})
                    update.return_value = RestObj({"updated": "thing"})
                    self.assertEqual(
                        _create_project("name", {}, "repo"), RestObj({"new": "project"})
                    )
                    project.assert_called_with(
                        "name", "repo", variables="variable", property="test"
                    )

    def test_create_project_update(self):
        with mock.patch("sasctl.tasks._format_properties") as format_properties:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.create_project"
            ) as project:
                with mock.patch(
                    "sasctl._services.model_repository.ModelRepository.update_project"
                ) as update:
                    format_properties.return_value = (
                        {"predictionVariable": "test"},
                        "variable",
                    )
                    project.return_value = RestObj({"new": "project"})
                    update.return_value = RestObj({"updated": "thing"})
                    self.assertEqual(
                        _create_project("name", {}, "repo"),
                        RestObj({"updated": "thing"}),
                    )
                    project.assert_called_with(
                        "name", "repo", variables="variable", predictionVariable="test"
                    )


class TestUpdateProperties(TestCase):
    headers = {"Content-Type": "application/vnd.sas.collection+json"}

    single_var = [
        {
            "name": "var_test",
            "role": "the_dice",
            "type": "writer",
            "level": "high",
            "length": "long",
        }
    ]

    second_var = [
        {
            "name": "var_two",
            "role": "in_the_deep",
            "type": "mine",
            "level": "low",
            "length": "short",
        }
    ]

    model = {
        "function": "Classification",
        "targetLevel": "Binary",
        "targetVariable": "BAD",
        "targetEvent": "1",
        "classTargetValues": "1, 0",
        "algorithm": "logistic",
    }

    def test_update_variables(self):
        with mock.patch("sasctl.tasks._format_properties") as format_properties:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.post"
            ) as post:
                with mock.patch(
                    "sasctl._services.model_repository.ModelRepository.update_project"
                ) as update:
                    with mock.patch(
                        "sasctl._services.model_repository.ModelRepository.get_project"
                    ) as get:
                        format_properties.return_value = ({}, self.single_var)
                        get.return_value = RestObj(id="123")
                        _update_properties("test", "model")
                        post.assert_called_with(
                            "projects/123/variables",
                            json=self.single_var,
                            headers=self.headers,
                        )

    def test_no_var_update(self):
        with mock.patch("sasctl.tasks._format_properties") as format_properties:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.post"
            ) as post:
                with mock.patch(
                    "sasctl._services.model_repository.ModelRepository.update_project"
                ) as update:
                    with mock.patch(
                        "sasctl._services.model_repository.ModelRepository.get_project"
                    ) as get:
                        format_properties.return_value = ({}, self.single_var)
                        get.return_value = RestObj(id="123", variables=self.single_var)
                        _update_properties("test", "model")
                        post.assert_not_called()

    def test_update_properties(self):
        with mock.patch("sasctl.tasks._format_properties") as format_properties:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.post"
            ) as post:
                with mock.patch(
                    "sasctl._services.model_repository.ModelRepository.update_project"
                ) as update:
                    with mock.patch(
                        "sasctl._services.model_repository.ModelRepository.get_project"
                    ) as get:
                        get.return_value = RestObj()
                        format_properties.return_value = (self.model, [])
                        _update_properties("test", "model")
                        update.assert_called_with(RestObj(self.model))
