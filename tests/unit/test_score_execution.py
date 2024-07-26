#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2024, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
import random
import tempfile
import unittest
import copy
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from requests import HTTPError

import sasctl.pzmm as pzmm
from sasctl import current_session
from sasctl.core import RestObj, VersionInfo
from sasctl._services.score_definitions import ScoreDefinitions as sd
from sasctl._services.score_execution import ScoreExecution as se


# Creating a CustomMock for list_executions to avoid a TypeError when comparing status code from mock with >= 400 in score_execution
class CustomMock:
    def __init__(self, status_code, json_info):
        self.status_code = status_code
        self.json_info = json_info

    def get(self, key1, key2=None, key3=None):
        if key2 is None and key3 is None:
            return self.json_info[key1]
        else:
            return self.json_info[key1][key2][key3]


def test_create_score_execution():
    """
    Test Cases:
    - Valid score definition id and invalid list_executions argument
    - Invalid score definition id
    - Valid list_executions argument with execution already running but invalid delete_execution argument
    - Valid list_executions argument with execution already running but valid delete_execution argument
    - Valid list_executions argument without execution already running
    - With output table specified within create_score_execution arguments

    """

    # Mocking a session to allow the post call to go through
    with mock.patch("sasctl.core.Session._get_authorization_token"):
        current_session("example.com", "username", "password")
    # Mocking what a successful score execution post would be if the user specified an output table
    TARGET = {"outputTable": {"tableName": "example_table"}}
    # Creating a target to compare post call responses to
    target = copy.deepcopy(TARGET)

    # Mocking the REST API calls and functions
    with mock.patch(
        "sasctl._services.score_definitions.ScoreDefinitions.get_definition"
    ) as get_definition:
        with mock.patch(
            "sasctl._services.score_execution.ScoreExecution.list_executions"
        ) as list_executions:
            with mock.patch(
                "sasctl._services.score_execution.ScoreExecution.delete_execution"
            ) as delete_execution:
                with mock.patch(
                    "sasctl._services.score_execution.ScoreExecution.post"
                ) as post:
                    # Invalid score definition id test case
                    get_definition.return_value.status_code = 404
                    with pytest.raises(HTTPError):
                        se.create_score_execution(score_definition_id="12345")

                    # Valid score definition id and invalid list_executions argument test case
                    get_definition.return_value.status_code = 200
                    get_definition.return_value.json.return_value = {
                        "inputData": {
                            "libraryName": "cas-shared-default",
                            "tableName": "test_table",
                        },
                        "name": "score_def_name",
                        "objectDescriptor": {
                            "name": "test_model",
                            "type": "sas.publish.example",
                            "uri": "/modelPublish/models/example",
                        },
                    }
                    list_executions.return_value.status_code = 404
                    with pytest.raises(HTTPError):
                        se.create_score_execution(score_definition_id="12345")

                    # Valid list_executions argument with execution already running but invalid delete_execution argument test case
                    list_mock_execution = CustomMock(
                        status_code=200,
                        json_info={"count": 1, "items": [{"id": "1234"}]},
                    )
                    list_executions.return_value = list_mock_execution
                    delete_execution.return_value.status_code = 404
                    with pytest.raises(HTTPError):
                        se.create_score_execution(score_definition_id="12345")

                    # Valid list_executions argument with execution already running but valid delete_execution argument test case
                    list_executions.return_value = list_mock_execution
                    delete_execution.return_value.status_code = 200
                    response = se.create_score_execution(score_definition_id="3456")
                    assert response

                    # Valid list_executions argument without execution already running test case
                    list_mock_execution_diff_count = CustomMock(
                        status_code=200,
                        json_info={"count": 0, "items": [{"id": "1234"}]},
                    )
                    list_executions.return_value = list_mock_execution_diff_count
                    response = se.create_score_execution(score_definition_id="12345")
                    assert response

                    # Checking whether the output table name remained the default empty string or the default changed as writted in score_execution
                    data = post.call_args
                    json_data = json.loads(data.kwargs["data"])
                    assert json_data["outputTable"]["tableName"] != ""

                    # With output table specified within create_score_execution arguments test case
                    response = se.create_score_execution(
                        score_definition_id="12345", output_table_name="example_table"
                    )
                    assert response
                    assert post.call_count == 3

                    # Checking whether specified output table name or the default output table name is in the response
                    data = post.call_args
                    json_data = json.loads(data.kwargs["data"])
                    assert (
                        target["outputTable"]["tableName"]
                        == json_data["outputTable"]["tableName"]
                    )
