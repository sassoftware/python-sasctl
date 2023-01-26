#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from unittest import mock

import pytest

from sasctl.core import RestObj
from sasctl._services import workflow


def test_list_workflow_prompt_invalidworkflow():

    WORKFLOWS = [
        {"name": "Test W", "id": "12345"},
        {
            "name": "TestW2",
            "id": "98765",
            "prompts": [
                {"id": "98765", "variableName": "projectId", "variableType": "string"}
            ],
        },
    ]
    wf = workflow.Workflow()

    with mock.patch(
        "sasctl._services.workflow.Workflow" ".list_definitions"
    ) as list_definitions:
        list_definitions.return_value = WORKFLOWS
        with pytest.raises(ValueError):
            # Project missing
            _ = wf.list_workflow_prompt("bad")


def test_list_workflow_prompt_workflownoprompt():

    WORKFLOWS = [
        {"name": "Test W", "id": "12345"},
        {
            "name": "TestW2",
            "id": "98765",
            "prompts": [
                {"id": "98765", "variableName": "projectId", "variableType": "string"}
            ],
        },
    ]
    wf = workflow.Workflow()

    with mock.patch(
        "sasctl._services.workflow.Workflow" ".list_definitions"
    ) as list_definitions:
        list_definitions.return_value = WORKFLOWS

        # Testing no prompts on workflow with name
        testresult = wf.list_workflow_prompt("Test W")
        print(testresult)
        assert testresult is None

        # Testing no prompts on workflow with id
        testresult = wf.list_workflow_prompt("12345")
        print(testresult)
        assert testresult is None


def test_list_workflow_prompt_workflowprompt():

    WORKFLOWS = [
        {"name": "Test W", "id": "12345"},
        {
            "name": "TestW2",
            "id": "98765",
            "prompts": [
                {"id": "98765", "variableName": "projectId", "variableType": "string"}
            ],
        },
    ]
    wf = workflow.Workflow()

    with mock.patch(
        "sasctl._services.workflow.Workflow" ".list_definitions"
    ) as list_definitions:
        list_definitions.return_value = WORKFLOWS

        # Testing workflow with id and prompts
        testresult = wf.list_workflow_prompt("TestW2")
        print(testresult)
        assert testresult is not None

        # Testing workflow with id and prompts
        testresult = wf.list_workflow_prompt("98765")
        print(testresult)
        assert testresult is not None


@mock.patch("sasctl._services.workflow.Workflow.post")
@mock.patch("sasctl._services.workflow.Workflow._find_specific_workflow")
def test_run_workflow_definition_no_prompts(get_workflow, post):
    """Verify correct REST call to run a workflow with no inputs."""

    DEFINITION_ID = "abc-123"

    get_workflow.return_value = RestObj(name="Inquisition", id=DEFINITION_ID)
    post.return_value.status_code = 200
    post.return_value.json.return_value = {}

    workflow.Workflow.run_workflow_definition("inquisition")

    assert post.call_count == 1
    url, params = post.call_args
    assert url[0].startswith("/processes")
    assert DEFINITION_ID in url[0]
    assert "json" not in params  # should not have passed any prompt info


@mock.patch("sasctl._services.workflow.Workflow.post")
@mock.patch("sasctl._services.workflow.Workflow._find_specific_workflow")
def test_run_workflow_definition_with_prompts(get_workflow, post):
    """Verify correct REST call and prompt formatting."""

    # Mock response from looking up workflow information.
    # NOTE: needs to include prompt information to match with prompt input values.
    WORKFLOW = RestObj(
        {
            "name": "Inquisition",
            "id": "abc-123",
            "prompts": [
                {"variableName": "moto", "variableType": "string", "version": 3},
                {"variableName": "date", "variableType": "dateTime", "version": 1},
            ],
        }
    )

    PROMPTS = {
        "moto": "No one expects the Spanish Inquisition!",
        "date": datetime.datetime(1912, 1, 1, 17, 30),
    }

    # Return mock workflow when asked
    get_workflow.return_value = WORKFLOW

    # Return a valid HTTP response for POSTs
    post.return_value.status_code = 200
    post.return_value.json.return_value = {}

    workflow.Workflow.run_workflow_definition("inquisition", prompts=PROMPTS)

    assert post.call_count == 1
    url, params = post.call_args

    assert url[0].startswith("/processes")
    assert WORKFLOW["id"] in url[0]

    # Check each prompt value that was passed and ensure it was correctly
    # matched to the prompts defined by the workflow.
    for name, value in PROMPTS.items():

        # Find the matching variable entry in the POST data
        variable = next(v for v in params["json"]["variables"] if v["name"] == name)

        # String and datetime prompts should be passed as string values
        if isinstance(value, (str, datetime.datetime)):
            assert isinstance(variable["value"], str)

        if isinstance(value, datetime.datetime):
            assert variable["type"] == "dateTime"
