#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from sasctl._services import workflow


def test_list_workflow_prompt_invalidworkflow():

    WORKFLOWS = [
        {'name': 'Test W', 'id': '12345'},
        {
            'name': 'TestW2',
            'id': '98765',
            'prompts': [
                {'id': '98765', 'variableName': 'projectId', 'variableType': 'string'}
            ],
        },
    ]
    wf = workflow.Workflow()

    with mock.patch(
        'sasctl._services.workflow.Workflow' '.list_definitions'
    ) as list_definitions:
        list_definitions.return_value = WORKFLOWS
        with pytest.raises(ValueError):
            # Project missing
            _ = wf.list_workflow_prompt('bad')


def test_list_workflow_prompt_workflownoprompt():

    WORKFLOWS = [
        {'name': 'Test W', 'id': '12345'},
        {
            'name': 'TestW2',
            'id': '98765',
            'prompts': [
                {'id': '98765', 'variableName': 'projectId', 'variableType': 'string'}
            ],
        },
    ]
    wf = workflow.Workflow()

    with mock.patch(
        'sasctl._services.workflow.Workflow' '.list_definitions'
    ) as list_definitions:
        list_definitions.return_value = WORKFLOWS

        # Testing no prompts on workflow with name
        testresult = wf.list_workflow_prompt('Test W')
        print(testresult)
        assert testresult is None

        # Testing no prompts on workflow with id
        testresult = wf.list_workflow_prompt('12345')
        print(testresult)
        assert testresult is None


def test_list_workflow_prompt_workflowprompt():

    WORKFLOWS = [
        {'name': 'Test W', 'id': '12345'},
        {
            'name': 'TestW2',
            'id': '98765',
            'prompts': [
                {'id': '98765', 'variableName': 'projectId', 'variableType': 'string'}
            ],
        },
    ]
    wf = workflow.Workflow()

    with mock.patch(
        'sasctl._services.workflow.Workflow' '.list_definitions'
    ) as list_definitions:
        list_definitions.return_value = WORKFLOWS

        # Testing workflow with id and prompts
        testresult = wf.list_workflow_prompt('TestW2')
        print(testresult)
        assert testresult is not None

        # Testing workflow with id and prompts
        testresult = wf.list_workflow_prompt('98765')
        print(testresult)
        assert testresult is not None
