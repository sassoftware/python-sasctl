#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import datetime
from unittest import mock

import pytest
from sasctl import current_session
from sasctl.services import model_repository as mr


def test_create_model():

    MODEL_NAME = "Test Model"
    PROJECT_NAME = "Test Project"
    PROJECT_ID = "12345"
    USER = "username"

    with mock.patch("sasctl.core.Session._get_authorization_token"):
        current_session("example.com", USER, "password")

    TARGET = {
        "name": MODEL_NAME,
        "projectId": PROJECT_ID,
        "modeler": USER,
        "description": "model description",
        "function": "Classification",
        "algorithm": "Dummy Algorithm",
        "tool": "pytest",
        "champion": True,
        "role": "champion",
        "immutable": True,
        "retrainable": True,
        "scoreCodeType": None,
        "targetVariable": None,
        "trainTable": None,
        "classificationEventProbabilityVariableName": None,
        "classificationTargetEventValue": None,
        "location": None,
        "properties": [
            {"name": "custom1", "value": 123, "type": "numeric"},
            {"name": "custom2", "value": "somevalue", "type": "string"},
            # {'name': 'customDate', 'value': 1672462800000, 'type': 'date'},
            {"name": "customDateTime", "value": 1672481272000, "type": "dateTime"},
        ],
        "inputVariables": [],
        "outputVariables": [],
        "version": 2,
    }

    # Passed params should be set correctly
    target = copy.deepcopy(TARGET)
    with mock.patch(
        "sasctl._services.model_repository.ModelRepository.get_project"
    ) as get_project:
        with mock.patch(
            "sasctl._services.model_repository.ModelRepository" ".get_model"
        ) as get_model:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.post"
            ) as post:
                get_project.return_value = {"id": PROJECT_ID}
                get_model.return_value = None
                _ = mr.create_model(
                    MODEL_NAME,
                    PROJECT_NAME,
                    description=target["description"],
                    function=target["function"],
                    algorithm=target["algorithm"],
                    tool=target["tool"],
                    is_champion=True,
                    is_immutable=True,
                    is_retrainable=True,
                    properties=dict(
                        custom1=123,
                        custom2="somevalue",
                        # customDate=datetime.date(2022, 12, 31),
                        customDateTime=datetime.datetime(
                            2022, 12, 31, 10, 7, 52, tzinfo=datetime.timezone.utc
                        ),
                    ),
                )
                assert post.call_count == 1
            url, data = post.call_args

            # dict isn't guaranteed to preserve order so k/v pairs of properties=dict()
            # may be returned in a different order
            assert sorted(target["properties"], key=lambda d: d["name"]) == sorted(
                data["json"]["properties"], key=lambda d: d["name"]
            )

            target.pop("properties")
            data["json"].pop("properties")
            assert target == data["json"]

    # Model dict w/ parameters already specified should be allowed
    # Explicit overrides should be respected.
    target = copy.deepcopy(TARGET)
    with mock.patch(
        "sasctl._services.model_repository.ModelRepository.get_project"
    ) as get_project:
        with mock.patch(
            "sasctl._services.model_repository.ModelRepository" ".get_model"
        ) as get_model:
            with mock.patch(
                "sasctl._services.model_repository.ModelRepository.post"
            ) as post:
                get_project.return_value = {"id": PROJECT_ID}
                get_model.return_value = None
                _ = mr.create_model(
                    copy.deepcopy(target), PROJECT_NAME, description="Updated Model"
                )
            target["description"] = "Updated Model"
            assert post.call_count == 1
            url, data = post.call_args

            # dicts don't preserve order so property order may not match
            assert target["properties"] == data["json"]["properties"]
            target.pop("properties")
            data["json"].pop("properties")
            assert target == data["json"]


def test_copy_analytic_store():
    # Create a dummy session
    with mock.patch("sasctl.core.Session._get_authorization_token"):
        current_session("example.com", "user", "password")

    MODEL_ID = 12345
    # Intercept calls to lookup the model & call the "copyAnalyticStore" link
    with mock.patch(
        "sasctl._services.model_repository.ModelRepository" ".get_model"
    ) as get_model:
        with mock.patch(
            "sasctl._services.model_repository.ModelRepository" ".request_link"
        ) as request_link:

            # Return a dummy Model with a static id
            get_model.return_value = {"id": MODEL_ID}
            mr.copy_analytic_store(MODEL_ID)

            # Should have been called once by default, and again to refresh
            # links
            assert get_model.call_count == 2
            assert request_link.call_count == 1

            args, _ = request_link.call_args
            obj, rel = args
            assert obj == get_model.return_value
            assert rel == "copyAnalyticStore"


def test_get_model_by_name():
    """If multiple models exist with the same name, a warning should be raised.

    From https://github.com/sassoftware/python-sasctl/issues/92
    """

    MODEL_NAME = "Test Model"

    # Create a dummy session
    with mock.patch("sasctl.core.Session._get_authorization_token"):
        current_session("example.com", "user", "password")

    mock_responses = [
        # First response is for list_items/list_models
        [{"id": 12345, "name": MODEL_NAME}, {"id": 67890, "name": MODEL_NAME}],
        # Second response is mock GET for model details
        {"id": 12345, "name": MODEL_NAME},
    ]

    with mock.patch(
        "sasctl._services.model_repository.ModelRepository.request"
    ) as request:
        request.side_effect = mock_responses

        with pytest.warns(Warning):
            result = mr.get_model(MODEL_NAME)
    assert result["id"] == 12345
    assert result["name"] == MODEL_NAME


def test_add_model_content():

    with mock.patch(
        "sasctl._services.model_repository.ModelRepository.get_model",
        return_value={"id": 123},
    ):
        with mock.patch(
            "sasctl._services.model_repository.ModelRepository.post"
        ) as post:
            text_data = "Test text file contents"

            # Basic upload of text data
            mr.add_model_content(None, text_data, "test.txt")
            assert post.call_args[1]["files"] == {
                "files": ("test.txt", text_data, "multipart/form-data")
            }

            # Upload of text data with content type
            mr.add_model_content(
                None, text_data, "test.txt", content_type="application/text"
            )
            assert post.call_args[1]["files"] == {
                "files": ("test.txt", text_data, "application/text")
            }

            # Upload of dict data without content type
            import json

            dict_data = {"data": text_data}
            mr.add_model_content(None, dict_data, "dict.json")
            assert post.call_args[1]["files"] == {
                "files": ("dict.json", json.dumps(dict_data), "multipart/form-data")
            }

            # Upload of binary data should include content type
            binary_data = "Test binary file contents".encode()
            mr.add_model_content(None, binary_data, "test.pkl")
            assert post.call_args[1]["files"] == {
                "files": ("test.pkl", binary_data, "application/octet-stream")
            }

            # Should be able to customize content type
            mr.add_model_content(
                None, binary_data, "test.pkl", content_type="application/image"
            )
            assert post.call_args[1]["files"] == {
                "files": ("test.pkl", binary_data, "application/image")
            }
