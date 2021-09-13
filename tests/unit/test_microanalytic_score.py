#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest import mock

from sasctl.services import microanalytic_score as mas

from sasctl import current_session
from sasctl.core import RestObj

with mock.patch('sasctl.core.Session.get_auth'):
    current_session('example.com', 'username', 'password')


def test_create_python_module():
    with mock.patch('sasctl.services.microanalytic_score.post') as post:
        with pytest.raises(ValueError):
            mas.create_module()  # Source code is required

    with mock.patch('sasctl.services.microanalytic_score.post') as post:
        source = '\n'.join(
            (
                "def testMethod(var1, var2):",
                "    'Output: out1, out2'",
                "    out1 = var1 + 5",
                "    out2 = var2.upper()",
                "    return out1, out2",
            )
        )
        mas.create_module(source=source)

        assert post.call_count == 1
        json = post.call_args[1].get('json', {})
        assert 'text/x-python' == json['type']
        assert 'public' == json['scope']


def test_delete_module(caplog):
    import logging

    caplog.set_level(logging.INFO, 'sasctl._services.service')

    # Delete should succeed even if object couldn't be found on server
    with mock.patch(
        'sasctl._services.microanalytic_score.MicroAnalyticScore' '.get'
    ) as get:
        get.return_value = None

        assert mas.delete_module('spam') is None
        assert any("Object 'spam' not found" in m for m in caplog.messages)


def test_define_steps():

    # Mock module to be returned
    module = RestObj(
        name='unittestmodule', id='unittestmodule', stepIds=['step1', 'step2']
    )

    # Mock module step with no inputs
    step1 = RestObj(id='post')

    # Mock module step with multiple inputs
    step2 = RestObj(
        {
            "id": "score",
            "inputs": [
                {"name": "age", "type": "decimal", "dim": 0, "size": 0},
                {"name": "b", "type": "decimal", "dim": 0, "size": 0},
                {"name": "chas", "type": "decimal", "dim": 0, "size": 0},
                {"name": "crim", "type": "decimal", "dim": 0, "size": 0},
                {"name": "dis", "type": "decimal", "dim": 0, "size": 0},
                {"name": "indus", "type": "decimal", "dim": 0, "size": 0},
                {"name": "lstat", "type": "decimal", "dim": 0, "size": 0},
                {"name": "nox", "type": "decimal", "dim": 0, "size": 0},
                {"name": "ptratio", "type": "decimal", "dim": 0, "size": 0},
                {"name": "rad", "type": "decimal", "dim": 0, "size": 0},
                {"name": "rm", "type": "decimal", "dim": 0, "size": 0},
                {"name": "tax", "type": "decimal", "dim": 0, "size": 0},
                {"name": "zn", "type": "decimal", "dim": 0, "size": 0},
            ],
            "outputs": [
                {"name": "em_prediction", "type": "decimal", "dim": 0, "size": 0},
                {"name": "p_price", "type": "decimal", "dim": 0, "size": 0},
                {"name": "_warn_", "type": "string", "dim": 0, "size": 4},
            ],
        }
    )

    with mock.patch(
        'sasctl._services.microanalytic_score.MicroAnalyticScore.get_module'
    ) as get_module:
        with mock.patch(
            'sasctl._services.microanalytic_score.MicroAnalyticScore' '.get_module_step'
        ) as get_step:
            get_module.return_value = module
            get_step.side_effect = [step1, step2]
            result = mas.define_steps(None)

    for step in get_step.side_effect:
        assert hasattr(result, step.id)


def test_define_steps_invalid_name():
    """Verify that invalid characters are stripped."""

    # Mock module to be returned
    # From bug reported by Paata
    module = RestObj(
        {
            'createdBy': 'paata',
            'creationTimeStamp': '2020-03-02T15:37:57.811Z',
            'id': 'petshop_model_xgb_new',
            'language': 'ds2',
            'links': [
                {
                    'href': '/microanalyticScore/modules',
                    'itemType': 'application/vnd.sas.microanalytic.module',
                    'method': 'GET',
                    'rel': 'up',
                    'type': 'application/vnd.sas.collection',
                    'uri': '/microanalyticScore/modules',
                },
                {
                    'href': '/microanalyticScore/modules/petshop_model_xgb_new',
                    'method': 'GET',
                    'rel': 'self',
                    'type': 'application/vnd.sas.microanalytic.module',
                    'uri': '/microanalyticScore/modules/petshop_model_xgb_new',
                },
                {
                    'href': '/microanalyticScore/modules/petshop_model_xgb_new/source',
                    'method': 'GET',
                    'rel': 'source',
                    'type': 'application/vnd.sas.microanalytic.module.source',
                    'uri': '/microanalyticScore/modules/petshop_model_xgb_new/source',
                },
                {
                    'href': '/microanalyticScore/modules/petshop_model_xgb_new/steps',
                    'itemType': 'application/vnd.sas.microanalytic.module.step',
                    'method': 'GET',
                    'rel': 'steps',
                    'type': 'application/vnd.sas.collection',
                    'uri': '/microanalyticScore/modules/petshop_model_xgb_new/steps',
                },
                {
                    'href': '/microanalyticScore/modules/petshop_model_xgb_new/submodules',
                    'itemType': 'application/vnd.sas.microanalytic.submodule',
                    'method': 'GET',
                    'rel': 'submodules',
                    'type': 'application/vnd.sas.collection',
                    'uri': '/microanalyticScore/modules/petshop_model_xgb_new/submodules',
                },
                {
                    'href': '/microanalyticScore/modules/petshop_model_xgb_new',
                    'method': 'PUT',
                    'rel': 'update',
                    'responseType': 'application/vnd.sas.microanalytic.module',
                    'type': 'application/vnd.sas.microanalytic.module',
                    'uri': '/microanalyticScore/modules/petshop_model_xgb_new',
                },
                {
                    'href': '/microanalyticScore/modules/petshop_model_xgb_new',
                    'method': 'DELETE',
                    'rel': 'delete',
                    'uri': '/microanalyticScore/modules/petshop_model_xgb_new',
                },
            ],
            'modifiedBy': 'paata',
            'modifiedTimeStamp': '2020-03-02T15:57:10.008Z',
            'name': '"petshop_model_XGB_new"',
            'properties': [
                {
                    'name': 'sourceURI',
                    'value': 'http://modelmanager/modelRepository/models/72facedf-8e36-418e-8145-1398686b997a',
                }
            ],
            'revision': 0,
            'scope': 'public',
            'stepIds': ['score'],
            'version': 2,
            'warnings': [],
        }
    )

    # Mock module step with multiple inputs
    step2 = RestObj(
        {
            "id": "score",
            "inputs": [
                {"name": "age", "type": "decimal", "dim": 0, "size": 0},
                {"name": "b", "type": "decimal", "dim": 0, "size": 0},
                {"name": "chas", "type": "decimal", "dim": 0, "size": 0},
                {"name": "crim", "type": "decimal", "dim": 0, "size": 0},
                {"name": "dis", "type": "decimal", "dim": 0, "size": 0},
                {"name": "indus", "type": "decimal", "dim": 0, "size": 0},
                {"name": "lstat", "type": "decimal", "dim": 0, "size": 0},
                {"name": "nox", "type": "decimal", "dim": 0, "size": 0},
                {"name": "ptratio", "type": "decimal", "dim": 0, "size": 0},
                {"name": "rad", "type": "decimal", "dim": 0, "size": 0},
                {"name": "rm", "type": "decimal", "dim": 0, "size": 0},
                {"name": "tax", "type": "decimal", "dim": 0, "size": 0},
                {"name": "zn", "type": "decimal", "dim": 0, "size": 0},
            ],
            "outputs": [
                {"name": "em_prediction", "type": "decimal", "dim": 0, "size": 0},
                {"name": "p_price", "type": "decimal", "dim": 0, "size": 0},
                {"name": "_warn_", "type": "string", "dim": 0, "size": 4},
            ],
        }
    )

    with mock.patch(
        'sasctl._services.microanalytic_score.MicroAnalyticScore.get_module'
    ) as get_module:
        with mock.patch(
            'sasctl._services.microanalytic_score.MicroAnalyticScore' '.get_module_step'
        ) as get_step:
            get_module.return_value = module
            get_step.side_effect = [step2]
            result = mas.define_steps(None)

    for step in get_step.side_effect:
        assert hasattr(result, step.id)
