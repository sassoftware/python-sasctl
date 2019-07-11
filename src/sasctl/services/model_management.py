#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from sasctl.core import post, _build_crud_funcs
from sasctl.services.model_publish import _publish_name

SERVICE_ROOT = '/modelManagement'


list_performance_definitions, get_performance_definition, \
update_performance_definition, delete_performance_definition = \
    _build_crud_funcs(SERVICE_ROOT + '/performanceTasks', 'performance task', 'performance tasks')


# TODO:  set ds2MultiType
def publish_model(model, destination, name=None, force=False):
    from .model_repository import get_model, get_model_link

    model_obj = get_model(model)

    if model_obj is None:
        model_name = model.name if hasattr(model, 'name') else str(model)
        raise ValueError("Model '{}' was not found.".format(model_name))

    model_uri = get_model_link(model_obj, 'self')

    # TODO: Verify allowed formats by destination type.
    # As of 19w04 MAS throws HTTP 500 if name is in invalid format.
    model_name = name or '{}_{}'.format(model_obj['name'].replace(' ', ''), model_obj['id']).replace('-', '')

    request = {
        "name": model_obj.get('name'),
        "notes": model_obj.get('description'),
        "modelContents": [
            {
                "modelName": _publish_name(model_obj.get('name')),
                "sourceUri": model_uri.get('uri'),
                "publishLevel": "model"
            }
        ],
        "destinationName": destination
    }

    # Publishes a model that has already been registered in the model repository.
    # Unlike model_publish service, does not require Code to be specified.
    r = post(SERVICE_ROOT + '/publish', json=request, params=dict(force=force), headers={'Content-Type': 'application/vnd.sas.models.publishing.request.asynchronous+json'})
    return r


def create_performance_definition(model, library_name, table_name, name=None, description=None, cas_server=None):
    from .model_repository import get_model, get_project

    model = get_model(model)
    project = get_project(model.projectId)

    # Performance data cannot be captured unless certain project properties have been configured.
    for required in ['targetVariable', 'targetLevel', 'predictionVariable']:
        if getattr(project, required, None) is None:
            raise ValueError("Project %s must have the '%s' property set." % (project.name, required))

    request = {'projectId': project.id,
               'modelIds': [model.id],
               'name': name or model.name + ' Performance',
               'description': description or 'Performance definition for model ' + model.name,
               'casServerId': cas_server or 'cas-shared-default',
               'resultLibrary': 'ModelPerformanceData',
               'dataLibrary': library_name,
               'dataTable': table_name
               }

    # If model doesn't specify input/output variables, try to pull from project definition
    if len(model.get('inputVariables', [])) > 0:
        request['inputVariables'] = [v.get('name') for v in model['inputVariables']]
        request['outputVariables'] = [v.get('name') for v in model['outputVariables']]
    else:
        request['inputVariables'] = [v.get('name') for v in project.get('variables', []) if v.get('role') == 'input']
        request['outputVariables'] = [v.get('name') for v in project.get('variables', []) if v.get('role') == 'output']

    return post(SERVICE_ROOT + '/performanceTasks', json=request, headers={'Content-Type': 'application/vnd.sas.models.performance.task+json'})