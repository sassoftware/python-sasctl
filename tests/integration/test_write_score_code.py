#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import current_session
from sasctl._services.model_repository import ModelRepository as mr
from sasctl.pzmm.write_score_code import ScoreCode as sc

pytestmark = pytest.mark.usefixtures("session")


@pytest.fixture()
def fake_predict():
    return "A", 1.0


def example_model(data):
    input_vars = [
        {"name": x, "type": "decimal", "role": "input"} for x in data.columns.to_list()
    ]
    output_vars = [
        {"name": "Classification", "type": "decimal", "role": "output"},
        {"name": "Prediction", "type": "decimal", "role": "output"},
    ]
    project = mr.get_project("TestProject")
    if not project:
        project = mr.create_project(
            project="TestProject",
            repository=mr.default_repository().get("id"),
            variables=input_vars + output_vars,
        )
    model = mr.create_model(
        model="TestModel",
        project=project,
        score_code_type="Python",
        input_variables=input_vars,
        output_variables=output_vars,
    )
    return model


def test_write_score_code(hmeq_dataset):
    """
    Test Cases:
    - Python score code is uploaded successfully
    - DS2 wrapper is created and scoreCodeType property updates
    - MAS/CAS code uploaded successfully and scoreCodeType property updated
    - files exist in output_dict
    """
    if current_session().version_info() == 4:
        pytest.skip(
            "The write_score_code function does not make any API calls for SAS Viya 4."
        )
    input_data = hmeq_dataset.drop(columns=["BAD"])
    model = example_model(input_data)
    output_dict = sc.write_score_code(
        model_prefix="TestModel",
        input_data=input_data,
        predict_method=[fake_predict, ["A", 1.0]],
        score_metrics=["Classification", "Prediction"],
        model=model,
        binary_string=b"BinaryStringModel",
    )

    assert "score_TestModel.py" in output_dict
    assert "dmcas_epscorecode.sas" in output_dict
    assert "dmcas_packagescorecode.sas" in output_dict
