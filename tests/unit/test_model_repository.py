#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from six.moves import mock

from sasctl import current_session
from sasctl.services import model_repository as mr


def test_create_model():

    MODEL_NAME = 'Test Model'
    PROJECT_NAME = 'Test Project'
    PROJECT_ID = '12345'
    USER = 'username'

    with mock.patch('sasctl.core.requests.Session.request'):
        current_session('example.com', USER, 'password')

    TARGET = {'name': MODEL_NAME,
              'projectId': PROJECT_ID,
              'modeler': USER,
              'description': 'model description',
              'function': 'Classification',
              'algorithm': 'Dummy Algorithm',
              'tool': 'pytest',
              'champion': True,
              'role': 'Champion',
              'properties': [{'name': 'custom1', 'value': 123},
                             {'name': 'custom2', 'value': 'somevalue'}]}

    # Passed params should be set correctly
    target = copy.deepcopy(TARGET)
    with mock.patch('sasctl._services.model_repository.ModelRepository.get_project') as get_project:
        with mock.patch('sasctl._services.model_repository.ModelRepository.post') as post:
            get_project.return_value = {'id': PROJECT_ID}
            _ = mr.create_model(MODEL_NAME,
                                PROJECT_NAME,
                                description=target['description'],
                                function=target['function'],
                                algorithm=target['algorithm'],
                                tool=target['tool'],
                                is_champion=target['champion'],
                                properties=dict(custom1=123, custom2='somevalue'))
            assert post.call_count == 1
            url, data = post.call_args

            # dict isn't guaranteed to preserve order
            # so k/v pairs of properties=dict() may be
            # returned in a different order
            assert sorted(target['properties'],
                          key=lambda d: d['name']) \
                   == sorted(data['json']['properties'],
                             key=lambda d: d['name'])

            target.pop('properties')
            data['json'].pop('properties')
            assert target == data['json']

    # Model dict w/ parameters already specified should be allowed
    # Explicit overrides should be respected.
    target = copy.deepcopy(TARGET)
    with mock.patch('sasctl._services.model_repository.ModelRepository.get_project') as get_project:
        with mock.patch('sasctl._services.model_repository.ModelRepository.post') as post:
            get_project.return_value = {'id': PROJECT_ID}
            _ = mr.create_model(copy.deepcopy(target), PROJECT_NAME, description='Updated Model')
            target['description'] = 'Updated Model'
            assert post.call_count == 1
            url, data = post.call_args

            # dicts don't preserve order so property order may not match
            assert target['properties'] == data['json']['properties']
            target.pop('properties')
            data['json'].pop('properties')
            assert target == data['json']


def test_copy_analytic_store():
    # Create a dummy session
    with mock.patch('sasctl.core.requests.Session.request'):
        current_session('example.com', 'user', 'password')

    MODEL_ID = 12345
    # Intercept calls to lookup the model & call the "copyAnalyticStore" link
    with mock.patch('sasctl._services.model_repository.ModelRepository'
                    '.get_model') as get_model:
        with mock.patch('sasctl._services.model_repository.ModelRepository'
                        '.request_link') as request_link:

            # Return a dummy Model with a static id
            get_model.return_value = {'id': MODEL_ID}
            mr.copy_analytic_store(MODEL_ID)

            # Should have been called once by default, and again to refresh
            # links
            assert get_model.call_count == 2
            assert request_link.call_count == 1

            args, _ = request_link.call_args
            obj, rel = args
            assert obj == get_model.return_value
            assert rel == 'copyAnalyticStore'
