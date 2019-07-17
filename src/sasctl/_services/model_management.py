#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service

from .. import services


class ModelManagement(Service):
    """The Model Management API provides basic resources for monitoring
    performance, comparing models, and running workflow processes.
    """
    _SERVICE_ROOT= '/modelManagement'


    list_performance_definitions, get_performance_definition, \
        update_performance_definition, delete_performance_definition = \
        Service._crud_funcs('/performanceTasks', 'performance task',
                            'performance tasks')

    # TODO:  set ds2MultiType
    def publish_model(self, model, destination, name=None, force=False):
        model_obj = services.model_repository.get_model(model)

        if model_obj is None:
            model_name = model.name if hasattr(model, 'name') else str(model)
            raise ValueError("Model '{}' was not found.".format(model_name))

        model_uri = services.model_repository.get_model_link(model_obj, 'self')

        # TODO: Verify allowed formats by destination type.
        # As of 19w04 MAS throws HTTP 500 if name is in invalid format.
        model_name = name or '{}_{}'.format(model_obj['name'].replace(' ', ''), model_obj['id']).replace('-', '')

        request = {
            "name": model_obj.get('name'),
            "notes": model_obj.get('description'),
            "modelContents": [
                {
                    "modelName": services.model_publish._publish_name(
                        model_obj.get('name')),
                    "sourceUri": model_uri.get('uri'),
                    "publishLevel": "model"
                }
            ],
            "destinationName": destination
        }

        # Publishes a model that has already been registered in the model
        # repository.
        # Unlike model_publish service, does not require Code to be specified.
        r = self.post('/publish', json=request, params=dict(force=force),
                      headers={'Content-Type': 'application/vnd.sas.models.publishing.request.asynchronous+json'})
        return r

    def create_performance_definition(self, model, library_name, table_name,
                                      name=None, description=None,
                                      outputLibrary=None, cas_server=None):
        """Create the performance task definition in the model project to
        monitor model performance.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the
            model.
        library_name : str
            The library containing the input data, default is 'Public'.
        table_name : str
            The name used for the performance data.
        name : str
            The name of the performance task, default is 'Performance'.
        description : str
            The description of the performance task, default is 'Performance
            monitoring for model' + model.name.
        cas_server : str
            The CAS Server for the monitoring task, default is
            'cas-shared-default'.
        championMonitored : bool
            Indicates to monitor the project champion model.
        challengerMonitored : bool
            Indicates to monitor challenger models.
        includeAllData : bool
            Indicates whether to run a performance job against all the data
            tables in a library.
        scoreExecutionRequired : bool
            Indicates whether the scoring task execution is required. This
            should be set 'False' if you have provided the scores and 'True'
            if not.
        maxBins : int
            The maximum bins number, default is 10.
        resultLibrary : str
            The performance output table library, default is
            'ModelPerformanceData'.
        traceOn : bool
            Indicates whether to turn on tracing.
        performanceResultSaved : bool
            Indicates whether the performance results are saved.
        loadPerformanceResult : bool
            Indicates to load performance result data.

        Returns
        -------
        str
            Performance task definition schema in JSON format.

        """
        model = services.model_repository.get_model(model)
        project = services.model_repository.get_project(model.projectId)

        # Performance data cannot be captured unless certain project properties
        # have been configured.
        for required in ['targetVariable', 'targetLevel', 'predictionVariable']:
            if getattr(project, required, None) is None:
                raise ValueError("Project %s must have the '%s' property set." % (project.name, required))

        request = {'projectId': project.id,
                   'name': name or model.name + ' Performance',
                   'modelIds': [model.id],
                   'championMonitored': False,
                   'challengerMonitored': False,
                   'includeAllData': False,
                   'scoreExecutionRequired': False,
                   'maxBins': 10,
                   'resultLibrary': outputLibrary or 'ModelPerformanceData',
                   'traceOn': False,
                   'performanceResultSaved': True,
                   'dataLibrary': library_name or 'Public',
                   'loadPerformanceResult': False,
                   'description': description or 'Performance definition for model ' + model.name,
                   'casServerId': cas_server or 'cas-shared-default',
                   'dataPrefix': table_name
                   }

        # If model doesn't specify input/output variables, try to pull from
        # project definition
        if len(model.get('inputVariables', [])) > 0:
            request['inputVariables'] = [v.get('name') for v in model['inputVariables']]
            request['outputVariables'] = [v.get('name') for v in model['outputVariables']]
        else:
            request['inputVariables'] = [v.get('name') for v in project.get('variables', []) if v.get('role') == 'input']
            request['outputVariables'] = [v.get('name') for v in project.get('variables', []) if v.get('role') == 'output']

        return self.post('/performanceTasks', json=request,
                     headers={'Content-Type': 'application/vnd.sas.models.performance.task+json'})
