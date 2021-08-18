#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from sasctl import RestObj
from sasctl.services import model_management as mm


def test_create_performance_definition():
    import copy
    from sasctl import current_session

    PROJECT = RestObj({'name': 'Test Project', 'id': '98765'})
    MODEL = RestObj({'name': 'Test Model', 'id': '12345', 'projectId': PROJECT['id']})
    USER = 'username'

    with mock.patch('sasctl.core.Session.get_auth'):
        current_session('example.com', USER, 'password')

    with mock.patch(
        'sasctl._services.model_repository.ModelRepository' '.get_model'
    ) as get_model:
        with mock.patch(
            'sasctl._services.model_repository.ModelRepository' '.get_project'
        ) as get_project:
            with mock.patch(
                'sasctl._services.model_management.ModelManagement' '.post'
            ) as post:
                get_model.return_value = MODEL

                with pytest.raises(ValueError):
                    # Project missing all required properties
                    get_project.return_value = copy.deepcopy(PROJECT)
                    _ = mm.create_performance_definition(
                        'model', 'TestLibrary', 'TestData'
                    )

                with pytest.raises(ValueError):
                    # Project missing some required properties
                    get_project.return_value = copy.deepcopy(PROJECT)
                    get_project.return_value['targetVariable'] = 'target'
                    _ = mm.create_performance_definition(
                        'model', 'TestLibrary', 'TestData'
                    )

                with pytest.raises(ValueError):
                    # Project missing some required properties
                    get_project.return_value = copy.deepcopy(PROJECT)
                    get_project.return_value['targetLevel'] = 'interval'
                    _ = mm.create_performance_definition(
                        'model', 'TestLibrary', 'TestData'
                    )

                with pytest.raises(ValueError):
                    # Project missing some required properties
                    get_project.return_value = copy.deepcopy(PROJECT)
                    get_project.return_value['function'] = 'classification'
                    _ = mm.create_performance_definition(
                        'model', 'TestLibrary', 'TestData'
                    )

                with pytest.raises(ValueError):
                    # Project missing some required properties
                    get_project.return_value = copy.deepcopy(PROJECT)
                    get_project.return_value['function'] = 'prediction'
                    _ = mm.create_performance_definition(
                        'model', 'TestLibrary', 'TestData'
                    )

                get_project.return_value = copy.deepcopy(PROJECT)
                get_project.return_value['targetVariable'] = 'target'
                get_project.return_value['targetLevel'] = 'interval'
                get_project.return_value['predictionVariable'] = 'predicted'
                get_project.return_value['function'] = 'prediction'
                _ = mm.create_performance_definition(
                    'model',
                    'TestLibrary',
                    'TestData',
                    max_bins=3,
                    monitor_challenger=True,
                    monitor_champion=True,
                )

            assert post.call_count == 1
            url, data = post.call_args

            assert PROJECT['id'] == data['json']['projectId']
            assert MODEL['id'] in data['json']['modelIds']
            assert 'TestLibrary' == data['json']['dataLibrary']
            assert 'TestData' == data['json']['dataPrefix']
            assert 'cas-shared-default' == data['json']['casServerId']
            assert data['json']['name'] is not None
            assert data['json']['description'] is not None
            assert data['json']['maxBins'] == 3
            assert data['json']['championMonitored'] == True
            assert data['json']['challengerMonitored'] == True

    def test_table_prefix_format():
        with pytest.raises(ValueError):
            # Underscores should not be allowed
            _ = mm.create_performance_definition('model', 'TestLibrary', 'invalid_name')


def test_execute_model_workflow_definition_invalidworkflow():

    PROJECT = RestObj({'name': 'Test Project', 'id': '98765'})
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

    with mock.patch(
        'sasctl._services.workflow.Workflow' '.list_enabled_definitions'
    ) as list_workflow_enableddefinitions:
        with mock.patch(
            'sasctl._services.model_repository.ModelRepository' '.get_project'
        ) as get_project:
            list_workflow_enableddefinitions.return_value = WORKFLOWS
            get_project.return_value = PROJECT
            with pytest.raises(ValueError):
                # Project missing
                _ = mm.execute_model_workflow_definition('TestLibrary', 'badworkflow')
