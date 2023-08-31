#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from sasctl import current_session
from sasctl._services.model_repository import ModelRepository as mr
from sasctl.pzmm.import_model import ImportModel as im
from sasctl.pzmm.import_model import model_exists, project_exists

pytestmark = pytest.mark.usefixtures("session")


@pytest.fixture()
def fake_predict():
    pass


def test_project_exists():
    """
    Test Cases:
    - No response provided; non-uuid value given for project
    """
    project = project_exists("Test_Project")
    assert project.name == "Test_Project"


def test_model_exists():
    """
    Test Cases:
    - No models in target project
    - Single model with same name in project w/ one model; force is True
    - Single model with same name in project w/ multiple models; force is True
    """
    project = mr.get_project("Test_Project")
    if not project:
        project_exists("Test_Project")
    model_exists(project, "Test_Model")

    model = mr.create_model("Test_Model", project)
    model_exists(project, "Test_Model", True)
    assert mr.get_model("Test_Model") is None

    model = mr.create_model("Test_Model", project)
    _ = mr.create_model("Test_Model_Dummy", project)
    model_exists(project, "Test_Model", True)
    assert mr.get_model("Test_Model") is None


def test_import_model(hmeq_dataset):
    """
    Test Cases:
    - no score code
    - Viya 4
    - Viya 3.5
    """
    model_files = {
        "Test.json": {"Test": True, "TestNum": 1},
        "Other_Test.json": {"Other": None, "TestNum": 2},
        "ModelProperties.json": {"Model": "Test"},
        "inputVar.json": [
            {"name": "LOAN", "level": "interval", "type": "decimal", "length": 8},
            {
                "name": "MORTDUE",
                "level": "interval",
                "type": "decimal",
                "length": 8,
            },
            {"name": "VALUE", "level": "interval", "type": "decimal", "length": 8},
            {"name": "YOJ", "level": "interval", "type": "decimal", "length": 8},
            {"name": "DEROG", "level": "interval", "type": "decimal", "length": 8},
            {"name": "DELINQ", "level": "interval", "type": "decimal", "length": 8},
            {"name": "CLAGE", "level": "interval", "type": "decimal", "length": 8},
            {"name": "NINQ", "level": "interval", "type": "decimal", "length": 8},
            {"name": "CLNO", "level": "interval", "type": "decimal", "length": 8},
            {
                "name": "DEBTINC",
                "level": "interval",
                "type": "decimal",
                "length": 8,
            },
        ],
        "outputVar.json": [
            {
                "name": "Classification",
                "level": "nominal",
                "type": "string",
                "length": 1,
            },
            {
                "name": "Probability",
                "level": "interval",
                "type": "decimal",
                "length": 8,
            },
        ],
    }
    model, model_files = im.import_model(
        model_files, "No_Score_Model", "Test_Project", overwrite_model=True
    )
    assert model == mr.get_model(model)
    for file in mr.get_model_contents(model):
        assert file.name in [
            "Test.json",
            "Other_Test.json",
            "ModelProperties.json",
            "inputVar.json",
            "outputVar.json",
        ]

    input_data = hmeq_dataset.drop(columns=["BAD", "REASON", "JOB"])
    output_vars = ["Classification", "Probability"]
    if current_session().version_info() == 4:
        model, model_files = im.import_model(
            model_files,
            "Test_Model",
            "Test_Project",
            input_data=input_data,
            predict_method=[fake_predict, [int, int]],
            score_metrics=output_vars,
            binary_string=b"Test Binary String",
            overwrite_model=True,
        )
        assert model == mr.get_model(model)
        for file in mr.get_model_contents(model):
            assert file.name in [
                "Test.json",
                "Other_Test.json",
                "inputVar.json",
                "ModelProperties.json",
                "outputVar.json",
                "score_Test_Model.py",
            ]
    else:
        model, model_files = im.import_model(
            model_files,
            "Test_Model",
            "Test_Project",
            input_data=input_data,
            predict_method=[fake_predict, [int, int]],
            score_metrics=output_vars,
            binary_string=b"Test Binary String",
            overwrite_model=True,
        )
        assert model == mr.get_model(model)
        for file in mr.get_model_contents(model):
            file_list = [
                "Test.json",
                "Other_Test.json",
                "score_Test_Model.py",
                "ModelProperties.json",
                "inputVar.json",
                "outputVar.json",
                "score.sas",
                "dmcas_epscorecode.sas",
                "dmcas_packagescorecode.sas",
            ]
            assert file.name in file_list
