#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from sasctl import current_session
from sasctl.pzmm import import_model
from sasctl.core import PagedList, RestObj, VersionInfo
from sasctl.pzmm.import_model import ImportModel as im
from sasctl.pzmm.import_model import model_exists, project_exists


def _fake_predict(fake="ABC"):
    return list(fake)


@patch("sasctl.pzmm.ScoreCode.write_score_code")
@patch("sasctl._services.model_repository.ModelRepository.get_project")
@patch("sasctl._services.model_repository.ModelRepository.import_model_from_zip")
@patch.multiple(
    "sasctl.pzmm.import_model", project_exists=MagicMock(), model_exists=MagicMock()
)
def test_import_model(mock_import, mock_project, mock_score):
    """
    Test Cases:
    - mlflow models set pickle type
    - no score code generation
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

        with pytest.warns():
            model, return_files = im.import_model(
                model_files, "Test_Model", "Test_Project"
            )

        model, return_files = im.import_model(
            model_files,
            "Test_Model",
            "Test_Project",
            pd.DataFrame(data=[[1, 1]]),
            _fake_predict,
            ["C", "P"],
            mlflow_details={"serialization_format": "dill"},
        )
        _, _, kwargs = mock_score.mock_calls[0]

        assert ("pickle_type", "dill") in kwargs.items()
        assert isinstance(return_files, dict)

        mock_version.return_value = VersionInfo(3)
        model, return_files = im.import_model(
            model_files,
            "Test_Model",
            "Test_Project",
            pd.DataFrame(data=[[1, 1]]),
            _fake_predict,
            ["C", "P"],
        )
        assert isinstance(return_files, dict)

        tmp_dir = tempfile.TemporaryDirectory()
        _ = tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir=tmp_dir.name)
        model_files = Path(tmp_dir.name)
        mock_score.return_value = None

        model, return_files = im.import_model(
            model_files,
            "Test_Model",
            "Test_Project",
            pd.DataFrame(data=[[1, 1]]),
            _fake_predict,
            ["C", "P"],
        )
        assert not isinstance(return_files, dict)

        mock_version.return_value = VersionInfo(4)
        model, return_files = im.import_model(
            model_files,
            "Test_Model",
            "Test_Project",
            pd.DataFrame(data=[[1, 1]]),
            _fake_predict,
            ["C", "P"],
        )
        assert not isinstance(return_files, dict)


@patch("sasctl._services.service.Service.get")
@patch("sasctl._services.model_repository.ModelRepository.create_project")
@patch("sasctl._services.model_repository.ModelRepository.default_repository")
@patch("sasctl.pzmm.import_model.get_model_properties")
def test_project_exists(mock_model_props, _, mock_project, mock_get):
    """
    Test Cases:
    - Project not found:
        - warning
            - if uuid, raise UUID not found
            - else
                - model_files
                - model_files is None
    - else:
        - if overwrite, update_properties called
        - else, compare_properties called
    """
    mock_get.return_value = "abc_repo"

    # UUID provided, but project not found
    project = str(uuid4())
    with pytest.warns(
        UserWarning, match=f"No project with the name or UUID {project} was found."
    ):
        with pytest.raises(SystemError):
            project_exists(project)

    # Non-UUID provided, but project not found
    project = "Test_Project"
    mock_project.return_value = RestObj(name="Test_Project")
    with pytest.warns(
        UserWarning, match=f"No project with the name or UUID {project} was found."
    ):
        response = project_exists(project)
        assert response == RestObj(name="Test_Project")
        mock_model_props.return_value = ("Test_Project", "Input_Var", "Output_Var")
        with patch.object(
            import_model, "_create_project", return_value=RestObj(name="Test_Project")
        ):
            response = project_exists(project, None, ["target"], "path/to/model/files")
            assert response == RestObj(name="Test_Project")

    # Project exists
    project_response = RestObj(name="Test_Project")
    with patch.object(import_model, "_compare_properties", return_value=None):
        response = project_exists(
            project, project_response, ["target"], "path/to/model/files"
        )
        assert response == RestObj(name="Test_Project")

    with patch.object(
        import_model,
        "_update_properties",
        return_value=RestObj(name="New_Test_Project"),
    ):
        response = project_exists(
            project, project_response, ["target"], "path/to/model/files", True
        )
        assert response == RestObj(name="New_Test_Project")


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
    mock_project.return_value = RestObj(
        name="Test Project", id="abc123", latestVersion="Test Version"
    )
    mock_versions.return_value = [RestObj(name="Test Version", id="def456")]
    mock_get.return_value = []
    model_exists("Test_Project", "Test_Model", False)

    with pytest.raises(ValueError):
        mock_get.return_value = RestObj({"name": "Test_Model", "id": "ghi789"})
        model_exists("Test_Project", "Test_Model", False)

    model_exists("Test_Project", "Test_Model", True)
    mock_delete.assert_called_once()
    mock_delete.reset_mock()

    with pytest.raises(ValueError):
        mock_get.return_value = PagedList(
            RestObj(
                items=[
                    {"name": "Test_Model", "id": "ghi789"},
                    {"name": "Other_Model", "id": "jkl012"},
                ],
                count=2,
            )
        )
        model_exists("Test_Project", "Test_Model", False)

    model_exists("Test_Project", "Test_Model", True)
    mock_delete.assert_called_once()
