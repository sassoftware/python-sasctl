#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from uuid import uuid4
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sasctl import current_session
from sasctl.core import VersionInfo, RestObj
from sasctl.pzmm.import_model import ImportModel as im
from sasctl.pzmm.import_model import project_exists, model_exists


def _fake_predict():
    pass


@patch("sasctl.pzmm.ScoreCode.write_score_code")
@patch("sasctl._services.model_repository.ModelRepository.get_project")
@patch("sasctl.pzmm.import_model.project_exists")
@patch("sasctl.pzmm.import_model.model_exists")
@patch("sasctl._services.model_repository.ModelRepository.import_model_from_zip")
def test_import_model(mock_import, m, p, mock_project, mock_score):
    """
    Test Cases:
    - mlflow models set pickle type
    - viya 4
        - in memory files
        - disk files
        - error for failed upload
    - viya 3.5
        - in memory files
        - disk files
        - error for failed upload
    """
    with patch("sasctl.core.Session._get_authorization_token"):
        current_session("example.com", "user", "password")

    with patch("sasctl.core.Session.version_info") as mock_version:
        mock_version.return_value = VersionInfo(4)
        mock_project.return_value = None
        mock_score.return_value = {
            "Test_Model_score.py": f"import sasctl\ndef score():\n{'':4}return \"Test\""
        }
        mock_import.return_value = {"name": "Test_Model", "id": "abcdef"}

        model_files = {
            "Test.json": json.dumps({"Test": True, "TestNum": 1}),
            "Other_Test.json": json.dumps({"Other": None, "TestNum": 2}),
        }
        return_files = im.import_model(
            model_files,
            "Test_Model",
            pd.DataFrame(data=[[1, 1]]),
            _fake_predict,
            ["C", "P"],
            "Test_Project",
            mlflow_details={"serialization_format": "dill"},
        )
        _, _, kwargs = mock_score.mock_calls[0]

        assert ("pickle_type", "dill") in kwargs.items()
        assert return_files

        mock_version.return_value = VersionInfo(3)
        return_files = im.import_model(
            model_files,
            "Test_Model",
            pd.DataFrame(data=[[1, 1]]),
            _fake_predict,
            ["C", "P"],
            "Test_Project",
        )
        assert return_files

        tmp_dir = tempfile.TemporaryDirectory()
        _ = tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir=tmp_dir.name)
        model_files = Path(tmp_dir.name)
        mock_score.return_value = None

        return_files = im.import_model(
            model_files,
            "Test_Model",
            pd.DataFrame(data=[[1, 1]]),
            _fake_predict,
            ["C", "P"],
            "Test_Project",
        )
        assert not return_files

        mock_version.return_value = VersionInfo(4)
        return_files = im.import_model(
            model_files,
            "Test_Model",
            pd.DataFrame(data=[[1, 1]]),
            _fake_predict,
            ["C", "P"],
            "Test_Project",
        )
        assert not return_files


@patch("sasctl._services.service.Service.get")
@patch("sasctl._services.model_repository.ModelRepository.create_project")
@patch("sasctl._services.model_repository.ModelRepository.default_repository")
def test_project_exists(r, mock_project, mock_get):
    """
    Test Cases:
    - Normal response given
    - Invalid UUID given
    - New project created
    """
    project = {"Name": "Test_Project"}
    response = project_exists(project, "Test_Project")
    assert response == project

    with pytest.raises(SystemError):
        response = project_exists(None, str(uuid4()))

    mock_get.return_value = "abc_repo"
    mock_project.return_value = RestObj(name="Test_Project")
    response = project_exists(None, "Test_Project")
    assert RestObj(name="Test_Project") == response


@patch("sasctl._services.model_repository.ModelRepository.delete_model")
@patch("sasctl._services.model_repository.ModelRepository.get")
@patch("sasctl._services.model_repository.ModelRepository.list_project_versions")
@patch("sasctl._services.model_repository.ModelRepository.get_project")
def test_model_exists(mock_project, mock_versions, mock_get, mock_delete):
    """
    Test Cases:
    - "Latest" project version
    - No models in version
    - One model in version; with same name; overwrite
    - <As above>; raise ValueError with no overwrite
    - > 1 model in version; with same name; overwrite
    - <As above>; raise ValueError with no overwrite
    """
    mock_project.return_value = {"id": "abc123", "latestVersion": "Test Version"}
    mock_versions.return_value = {{"name": "Test Version", "id": "def456"}}
    mock_get.return_value = []
    test = model_exists("Test_Project", "Test_Model", False)
    assert test is None

    with pytest.raises(ValueError):
        mock_get.return_value = {"name": "Test_Model"}
        model_exists("Test_Project", "Test_Model", False)

    model_exists("Test_Project", "Test_Model", True)
    mock_delete.assert_called_once()

    with pytest.raises(ValueError):
        mock_get.return_value = [{"name": "Test_Model"}, {"name": "Other_Model"}]
        model_exists("Test_Project", "Test_Model", False)

    model_exists("Test_Project", "Test_Model", True)
    mock_delete.assert_called_once()
