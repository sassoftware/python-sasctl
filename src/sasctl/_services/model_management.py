#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service
from ..utils.decorators import experimental

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
    @classmethod
    def publish_model(cls,
                      model,
                      destination,
                      name=None,
                      force=False,
                      reload_model_table=False):
        """

        Parameters
        ----------
        model
        destination
        name
        force : bool, optional
            Whether to overwrite the model if it already exists in the
            publishing `destination`.
        reload_model_table : bool, optional
            Whether the model table in CAS should be reloaded.  Defaults to
            False.

        Returns
        -------

        """
        from .model_repository import ModelRepository
        from .model_publish import ModelPublish

        mr = ModelRepository()
        mp = ModelPublish()

        model_obj = mr.get_model(model)

        if model_obj is None:
            model_name = model.name if hasattr(model, 'name') else str(model)
            raise ValueError("Model '{}' was not found.".format(model_name))

        model_uri = mr.get_model_link(model_obj, 'self')

        # TODO: Verify allowed formats by destination type.
        # As of 19w04 MAS throws HTTP 500 if name is in invalid format.
        model_name = name or '{}_{}'.format(model_obj['name'].replace(' ', ''),
                                            model_obj['id']).replace('-', '')

        request = {
            "name": model_obj.get('name'),
            "notes": model_obj.get('description'),
            "modelContents": [
                {
                    "modelName": mp._publish_name(
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
        r = cls.post('/publish',
                     json=request,
                     params=dict(force=force,
                                  reloadModelTable=reload_model_table),
                     headers={'Content-Type':
                                   'application/vnd.sas.models.publishing.request.asynchronous+json'})
        return r

    @classmethod
    def create_performance_definition(cls,
                                      model,
                                      library_name,
                                      table_prefix,
                                      name=None,
                                      description=None,
                                      monitor_champion=False,
                                      monitor_challenger=False,
                                      max_bins=None,
                                      scoring_required=False,
                                      all_data=False,
                                      save_output=True,
                                      output_library=None,
                                      autoload_output=False,
                                      cas_server=None,
                                      trace=False):
        """Create the performance task definition in the model project to
        monitor model performance.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the model.
        library_name : str
            The library containing the input data, default is 'Public'.
        table_prefix : str
            The name used for the performance data.
        name : str
            The name of the performance task.
        description : str
            The description of the performance task, default is 'Performance
            monitoring for model' + model.name.
        monitor_champion : bool
            Indicates to monitor the project champion model.
        monitor_challenger : bool
            Indicates to monitor challenger models.
        max_bins : int
            The maximum bins number, Must be >= 2.  Defaults to 10.
        scoring_required : bool
            Whether model scoring must be performed on the input data before
            performance results can be computed.  Should be `False` if target
            values are included in the `table_prefix` tables.
        all_data : bool
            Whether to run the performance job against all matching data tables
            in `library_name` or just the new tables.  Defaults to `False`.
        save_output : bool
            Whether to save the computed results to a table in `output_library`.
            Defaults to True.
        output_library : str
            Name of a CASLIB where computed results should be saved.  Defaults to
            'ModelPerformanceData'.
        autoload_output : bool
            Whether computed results should automatically be re-loaded
            after a CAS server restart.
        cas_server : str
            The CAS Server for the monitoring task, default is 'cas-shared-default'.
        trace : bool
            Whether to enable trace messages in the SAS job log when
            executing the performance definition.

        Returns
        -------
        RestObj
            The performance task definition schema

        """
        from .model_repository import ModelRepository

        if not scoring_required and '_' in table_prefix:
            raise ValueError(
                "Parameter 'table_prefix' cannot contain underscores."
                " Received a value of '%s'.") % table_prefix

        max_bins = 10 if max_bins is None else int(max_bins)
        if int(max_bins) < 2:
            raise ValueError("Parameter 'max_bins' must be at least 2.  "
                             "Received a value of '%s'." % max_bins)

        mr = ModelRepository()
        model = mr.get_model(model)
        project = mr.get_project(model.projectId)

        # Performance data cannot be captured unless certain project properties
        # have been configured.
        for required in ['targetVariable', 'targetLevel']:
            if getattr(project, required, None) is None:
                raise ValueError("Project %s must have the '%s' property set."
                                 % (project.name, required))
        if project.get('function') == 'classification' \
                and project.get('eventProbabilityVariable') is None:
            raise ValueError("Project %s must have the "
                             "'eventProbabilityVariable' property set."
                             % project.name)
        if project.get('function') == 'prediction' \
                and project.get('predictionVariable') is None:
            raise ValueError("Project %s must have the 'predictionVariable' "
                             "property set." % project.name)

        request = {'projectId': project.id,
                   'name': name or model.name + ' Performance',
                   'modelIds': [model.id],
                   'championMonitored': monitor_champion,
                   'challengerMonitored': monitor_challenger,
                   'maxBins': max_bins,
                   'resultLibrary': output_library or 'ModelPerformanceData',
                   'includeAllData': all_data,
                   'scoreExecutionRequired': scoring_required,
                   'performanceResultSaved': save_output,
                   'loadPerformanceResult': autoload_output,
                   'dataLibrary': library_name or 'Public',
                   'description': description or 'Performance definition for model ' + model.name,
                   'casServerId': cas_server or 'cas-shared-default',
                   'dataPrefix': table_prefix,
                   'traceOn': trace
                   }

        # If model doesn't specify input/output variables, try to pull from project definition
        if model.get('inputVariables', []):
            request['inputVariables'] = [v.get('name') for v in
                                         model['inputVariables']]
            request['outputVariables'] = [v.get('name') for v in
                                          model['outputVariables']]
        else:
            request['inputVariables'] = [v.get('name') for v in
                                         project.get('variables', []) if
                                         v.get('role') == 'input']
            request['outputVariables'] = [v.get('name') for v in
                                          project.get('variables', []) if
                                          v.get('role') == 'output']

        return cls.post('/performanceTasks', json=request,
                        headers={
            'Content-Type': 'application/vnd.sas.models.performance.task+json'})

    @classmethod
    def execute_performance_definition(cls, definition):
        """Launches a job to run a performance definition.

        Parameters
        ----------
        definition : str or dict
            The id or dictionary representation of a performance definition.

        Returns
        -------
        RestObj
            The executing job

        """
        definition = cls.get_performance_definition(definition)

        return cls.post('/performanceTasks/%s' % definition.id)

    @classmethod
    @experimental
    def list_model_workflow_definition(cls):
        """List all enabled Workflow Processes to execute on Model Project.

        Returns
        -------
        RestObj
            The list of workflows

        """
        from .workflow import Workflow
        wf = Workflow()

        return wf.list_enabled_definitions()

    @classmethod
    @experimental
    def list_model_workflow_prompt(cls, workflowName):
        """List prompt Workflow Processes Definitions.

        Parameters
        ----------
        workflowName : str
            Name or ID of an enabled workflow to retrieve inputs

        Returns
        -------
        list
            The list of prompts for specific workflow
    
        """
        from .workflow import Workflow
        wf = Workflow()
        
        return wf.list_workflow_prompt(workflowName)

    @classmethod
    @experimental
    def list_model_workflow_executed(cls, projectName):
        """List prompt Workflow Processes Definitions.

        Parameters
        ----------
        projectName : str
            Name of the Project list executed workflow

        Returns
        -------
        RestObj
            List of workflows associated to project

        """
        from .model_repository import ModelRepository
        mr = ModelRepository()

        project = mr.get_project(projectName)

        return cls.get('/workflowProcesses?filter=eq(associations.solutionObjectId,%22' + project['id'] + '%22)')

    @classmethod
    @experimental
    def execute_model_workflow_definition(cls, project_name, workflow_name, input=None):
        """Runs specific Workflow Processes Definitions.

        Parameters
        ----------
        project_name : str
            Name of the Project that will execute workflow
        workflow_name : str
            Name or ID of an enabled workflow to execute
        input : dict, optional
            Input values for the workflow for initial workflow prompt

        Returns
        -------
        RestObj
            The executing workflow
            
        """
        from .model_repository import ModelRepository
        from .workflow import Workflow

        mr = ModelRepository()
        wf = Workflow()
        
        project = mr.get_project(project_name)

        workflow = wf.run_workflow_definition(workflow_name, input=input)
        
        # Associations running workflow to model project, note workflow has to be running
        # THINK ABOUT: do we do a check on status of the workflow to determine if it is still running before associating?

        input = {"processName": workflow['name'],
                 "processId": workflow['id'],
                 "objectType": "MM_Project",
                 "solutionObjectName": project_name,
                 "solutionObjectId": project['id'],
                 "solutionObjectUri": "/modelRepository/projects/" + project['id'],
                 "solutionObjectMediaType": "application/vnd.sas.models.project+json"}
        
        #Note, you can get a HTTP Error 404: {"errorCode":74052,"message":"The workflow process for id <> cannot be found.
        #                                    Associations can only be made to running processes.","details":["correlator:
        #                                    e62c5562-2b11-45db-bcb7-933200cb0f0a","traceId: 3118c0fb1eb9702d","path:
        #                                    /modelManagement/workflowAssociations"],"links":[],"version":2,"httpStatusCode":404} 
        # Which is fine and expected like the Visual Experience.
        return cls.post('/workflowAssociations',
                        json=input,
                        headers={'Content-Type': 'application/vnd.sas.workflow.object.association+json'})


