#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from six.moves import mock

import sasctl.services.model_repository as mr


def test_create_model():
    import copy
    from sasctl import current_session

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
    with mock.patch('sasctl.services.model_repository.get_project') as get_project:
        with mock.patch('sasctl.services.model_repository.post') as post:
            get_project.return_value = {'id': PROJECT_ID}
            _ = mr.create_model(MODEL_NAME,
                                PROJECT_NAME,
                                description=target['description'],
                                function=target['function'],
                                algorithm=target['algorithm'],
                                tool=target['tool'],
                                is_champion=target['champion'],
                                properties=dict(custom1=123, custom2='somevalue'))
            post.assert_called_once()
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
    with mock.patch('sasctl.services.model_repository.get_project') as get_project:
        with mock.patch('sasctl.services.model_repository.post') as post:
            get_project.return_value = {'id': PROJECT_ID}
            _ = mr.create_model(copy.deepcopy(target), PROJECT_NAME, description='Updated Model')
            target['description'] = 'Updated Model'
            post.assert_called_once()
            url, data = post.call_args

            # dicts don't preserve order so property order may not match
            assert target['properties'] == data['json']['properties']
            target.pop('properties')
            data['json'].pop('properties')
            assert target == data['json']

