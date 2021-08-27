#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from sasctl.core import RestObj
from sasctl._services.model_repository import ModelRepository


def test_sklearn_metadata():
    pytest.importorskip('sklearn')

    from sasctl.tasks import _sklearn_to_dict
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    info = _sklearn_to_dict(LinearRegression())
    assert info['algorithm'] == 'Linear regression'
    assert info['function'] == 'prediction'

    info = _sklearn_to_dict(LogisticRegression())
    assert info['algorithm'] == 'Logistic regression'
    assert info['function'] == 'classification'

    info = _sklearn_to_dict(SVC())
    assert info['algorithm'] == 'Support vector machine'
    assert info['function'] == 'classification'

    info = _sklearn_to_dict(GradientBoostingClassifier())
    assert info['algorithm'] == 'Gradient boosting'
    assert info['function'] == 'classification'

    info = _sklearn_to_dict(DecisionTreeClassifier())
    assert info['algorithm'] == 'Decision tree'
    assert info['function'] == 'classification'

    info = _sklearn_to_dict(RandomForestClassifier())
    assert info['algorithm'] == 'Forest'
    assert info['function'] == 'classification'


def test_parse_module_url():
    from sasctl.tasks import _parse_module_url

    body = RestObj(
        {
            'createdBy': 'sasdemo',
            'creationTimeStamp': '2019-08-26T15:16:42.900Z',
            'destinationName': 'maslocal',
            'id': '62cae262-7287-412b-8f1d-bd2a12c8b434',
            'links': [
                {
                    'href': '/modelPublish/models/44d526bc-d513-4637-b8a7-72daee4a7730',
                    'method': 'GET',
                    'rel': 'up',
                    'type': 'application/vnd.sas.models.publishing.publish',
                    'uri': '/modelPublish/models/44d526bc-d513-4637-b8a7-72daee4a7730',
                },
                {
                    'href': '/modelPublish/models/44d526bc-d513-4637-b8a7-72daee4a7730/log',
                    'method': 'GET',
                    'rel': 'self',
                    'type': 'application/json',
                    'uri': '/modelPublish/models/44d526bc-d513-4637-b8a7-72daee4a7730/log',
                },
            ],
            'log': 'SUCCESS==={"links":[{"method":"GET","rel":"up","href":"/microanalyticScore/jobs","uri":"/microanalyticScore/jobs","type":"application/vnd.sas.collection","itemType":"application/vnd.sas.microanalytic.job"},{"method":"GET","rel":"self","href":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3","uri":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3","type":"application/vnd.sas.microanalytic.job"},{"method":"GET","rel":"source","href":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3/source","uri":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3/source","type":"application/vnd.sas.microanalytic.module.source"},{"method":"GET","rel":"submodules","href":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3/submodules","uri":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3/submodules","type":"application/vnd.sas.collection","itemType":"application/vnd.sas.microanalytic.submodule"},{"method":"DELETE","rel":"delete","href":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3","uri":"/microanalyticScore/jobs/465ecad8-cfd0-4403-ac8a-e49cd248fae3"},{"method":"GET","rel":"module","href":"/microanalyticScore/modules/decisiontree","uri":"/microanalyticScore/modules/decisiontree","type":"application/vnd.sas.microanalytic.module"}],"version":1,"createdBy":"sasdemo","creationTimeStamp":"2019-08-26T15:16:42.857Z","modifiedBy":"sasdemo","modifiedTimeStamp":"2019-08-26T15:16:48.988Z","id":"465ecad8-cfd0-4403-ac8a-e49cd248fae3","moduleId":"decisiontree","state":"completed","errors":[]}',
            'modelId': '459aae0d-d64f-4376-94e7-be31911f4bdb',
            'modelName': 'DecisionTree',
            'modifiedBy': 'sasdemo',
            'modifiedTimeStamp': '2019-08-26T15:16:49.315Z',
            'publishName': 'Decision Tree',
            'version': 1,
        }
    )

    msg = body.get('log').lstrip('SUCßCESS===')
    assert _parse_module_url(msg) == '/microanalyticScore/modules/decisiontree'


def test_save_performance_project_types():
    from sasctl.tasks import update_model_performance

    with mock.patch(
        'sasctl._services.model_repository.ModelRepository.get_model'
    ) as model:
        with mock.patch(
            'sasctl._services.model_repository.ModelRepository.get_project'
        ) as project:
            model.return_value = RestObj(name='fakemodel', projectId=1)

            # Function is required
            with pytest.raises(ValueError):
                project.return_value = {}
                update_model_performance(None, None, None)

            # Target Level is required
            with pytest.raises(ValueError):
                project.return_value = {'function': 'Prediction'}
                update_model_performance(None, None, None)

            # Prediction variable required
            with pytest.raises(ValueError):
                project.return_value = {
                    'function': 'Prediction',
                    'targetLevel': 'Binary',
                }
                update_model_performance(None, None, None)

            # Classification variable required
            with pytest.raises(ValueError):
                project.return_value = {
                    'function': 'classification',
                    'targetLevel': 'Binary',
                }
                update_model_performance(None, None, None)

    # Check projects w/ invalid properties


@mock.patch.object(ModelRepository, 'list_repositories')
@mock.patch.object(ModelRepository, 'get_project')
def test_register_model_403_error(get_project, list_repositories):
    """Verify HTTP 403 is converted to a user-friendly error.

    Depending on environment configuration, this can happen when attempting to
    find a repository.

    See: https://github.com/sassoftware/python-sasctl/issues/39
    """
    from urllib.error import HTTPError
    from sasctl.exceptions import AuthorizationError
    from sasctl.tasks import register_model

    get_project.return_value = {'name': 'Project Name'}
    list_repositories.side_effect = HTTPError(None, 403, None, None, None)

    # HTTP 403 error when getting repository should throw a user-friendly
    # AuthorizationError
    with pytest.raises(AuthorizationError):
        register_model(None, 'model name', 'project name')

    # All other errors should be bubbled up
    list_repositories.side_effect = HTTPError(None, 404, None, None, None)
    with pytest.raises(HTTPError):
        register_model(None, 'model name', 'project name')
