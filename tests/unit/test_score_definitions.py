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
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from requests import HTTPError

import sasctl.pzmm as pzmm
from sasctl import current_session
from sasctl.core import RestObj, VersionInfo, request
from sasctl._services.score_definitions import ScoreDefinitions as sd


def test_create_score_definition():
    """
    Test Cases:
    - Valid model id with input mapping and valid table name
    - Invalid model id
    - Valid table name without input mapping
    - Invalid table name with invalid file
    - Invalid table name with valid file and without input mapping

    """
    with mock.patch("sasctl.core.Session._get_authorization_token"):
        current_session("example.com", "username", "password")

    with mock.patch(
        "sasctl._services.model_repository.ModelRepository.get_model"
    ) as get_model:
        with mock.patch(
            "sasctl._services.cas_management.CASManagement.get_table"
        ) as get_table:
            with mock.patch(
                "sasctl._services.cas_management.CASManagement.upload_file"
            ) as upload_file:
                with mock.patch(
                    "sasctl._services.score_definitions.ScoreDefinitions.post"
                ) as post:
                    get_model.return_value.status_code = 404
                    with pytest.raises(HTTPError):
                        sd.create_score_definition(
                            score_def_name="test_create_sd",
                            model_id="12345",
                            table_name="test_table",
                        )

                    get_model.return_value.status_code = 200
                    get_model.return_value.json.return_value = {
                        "id": "12345",
                        "projectId": "54321",
                        "projectVersionId": "67890",
                        "name": "test_model",
                    }
                    get_table.return_value.status_code = 404
                    with pytest.raises(HTTPError):
                        sd.create_score_definition(
                            score_def_name="test_create_sd",
                            model_id="12345",
                            table_name="test_table",
                        )
                    get_table.return_value.status_code = 404
                    upload_file.return_value = None
                    get_table.return_value.status_code = 404
                    with pytest.raises(HTTPError):
                        sd.create_score_definition(
                            score_def_name="test_create_sd",
                            model_id="12345",
                            table_name="test_table",
                            table_file="test_path",
                        )

                    get_table.return_value.status_code = 404
                    upload_file.return_value = RestObj
                    get_table.return_value.status_code = 200
                    get_table.return_value.json.return_value = {
                        "tableName": "test_table"
                    }
                    response = sd.create_score_definition(
                        score_def_name="test_create_sd",
                        model_id="12345",
                        table_name="test_table",
                        table_file="test_path",
                    )
                    assert response

                    get_table.return_value.status_code = 200
                    get_table.return_value.json.return_value = {
                        "tableName": "test_table"
                    }
                    response = sd.create_score_definition(
                        score_def_name="test_create_sd",
                        model_id="12345",
                        table_name="test_table",
                        table_file="test_path",
                    )
                    assert response

                    get_model.return_value.status_code = 200
                    get_model.return_value.json.return_value = {
                        "id": "12345",
                        "projectId": "54321",
                        "projectVersionId": "67890",
                        "name": "test_model",
                        "inputVariables": [
                            {"name": "first"},
                            {"name": "second"},
                            {"name": "third"},
                        ],
                    }
                    get_table.return_value.status_code = 200
                    get_table.return_value.json.return_value = {
                        "tableName": "test_table"
                    }
                    response = sd.create_score_definition(
                        score_def_name="test_create_sd",
                        model_id="12345",
                        table_name="test_table",
                    )
                    assert response
                    assert post.call_count == 3
