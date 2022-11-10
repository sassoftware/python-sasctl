#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json

from .service import Service
from ..utils.decorators import experimental


class ModelManagement(Service):
    """The Model Management API provides basic resources for monitoring
    performance, comparing models, and running workflow processes.
    """

    _SERVICE_ROOT = "/modelManagement"

    (
        list_performance_definitions,
        get_performance_definition,
        update_performance_definition,
        delete_performance_definition,
    ) = Service._crud_funcs(
        "/performanceTasks", "performance task", "performance tasks"
    )

    # TODO:  set ds2MultiType
    @classmethod
    def publish_model(
        cls, model, destination, name=None, force=False, reload_model_table=False
    ):
        """

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the model.
        destination : str
            Name of destination to publish the model to.
        name : str, optional
            Provide a custom name for the published model. Defaults to None.
        force : bool, optional
            Whether to overwrite the model if it already exists in the
            publishing `destination`. Defaults to False.
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
            model_name = model.name if hasattr(model, "name") else str(model)
            raise ValueError("Model '{}' was not found.".format(model_name))

        # Include refresh in case get_model call does not return enough information
        model_uri = mr.get_model_link(model_obj, "self", True)

        # TODO: Verify allowed formats by destination type.
        # As of 19w04 MAS throws HTTP 500 if name is in invalid format.
        model_name = name or "{}_{}".format(
            model_obj["name"].replace(" ", ""), model_obj["id"]
        ).replace("-", "")

        request = {
            "name": model_obj.get("name"),
            "notes": model_obj.get("description"),
            "modelContents": [
                {
                    "modelName": mp._publish_name(model_name),
                    "sourceUri": model_uri.get("uri"),
                    "publishLevel": "model",
                }
            ],
            "destinationName": destination,
        }

        # Publishes a model that has already been registered in the model
        # repository.
        # Unlike model_publish service, does not require Code to be specified.
        r = cls.post(
            "/publish",
            json=request,
            params=dict(force=force, reloadModelTable=reload_model_table),
            headers={
                "Content-Type": "application/vnd.sas.models.publishing.request.asynchronous+json"
            },
        )
        return r

    @classmethod
    def create_performance_definition(
        cls,
        table_prefix,
        project=None,
        models=None,
        library_name="Public",
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
        trace=False,
    ):
        """Create the performance task definition in the model project to monitor model performance. In order to
        execute the performance task definition, run the execute_performance_definition function with the output from
        this function.

        Note that the performance task definition cannot be created if the project is not a valid form of model, or if
        targetVariable or targetLevel are not defined.

        Parameters
        ----------
        table_prefix : str
            The name used for the prefix of the performance data.
        project : str, optional
            Name or ID of the project. If no project is specified, it is inferred from the models argument. Defaults to
            None.
        models : str, list, or dict, optional
            The name or id of the model(s), or a dictionary representation of the model(s). For
            multiple models, input a list of model names, or a list of dictionaries. If no models are specified, all
            models in the project specified will be used. Defaults to None.
        library_name : str
            The library containing the input data, default is 'Public'.
        name : str, optional
            The name of the performance task, default is None.
        description : str, optional
            The description of the performance task, default is None.
        monitor_champion : bool, optional
            Indicates to monitor the project champion model, default is None.
        monitor_challenger : bool, optional
            Indicates to monitor challenger models, default is None.
        max_bins : int, optional
            The maximum bins number. Must be >= 2.  Defaults to 10.
        scoring_required : bool, optional
            Whether model scoring must be performed on the input data before performance results can be computed.
            Should be `False` if target values are included in the `table_prefix` tables. Defaults to `False`.
        all_data : bool, optional
            Whether to run the performance job against all matching data tables
            in `library_name` or just the new tables.  Defaults to `False`.
        save_output : bool, optional
            Whether to save the computed results to a table in `output_library`. Defaults to True.
        output_library : str, optional
            Name of a CASLIB where computed results should be saved.  Defaults to 'ModelPerformanceData'.
        autoload_output : bool, optional
            Whether computed results should automatically be re-loaded after a CAS server restart, defaults to False.
        cas_server : str, optional
            The CAS Server for the monitoring task, default is 'cas-shared-default'.
        trace : bool, optional
            Whether to enable trace messages in the SAS job log when executing the performance definition, defaults to
            False.

        Returns
        -------
        RestObj
            The performance task definition schema

        """
        from .model_repository import ModelRepository as mr

        if not scoring_required and "_" in table_prefix:
            raise ValueError(
                "Parameter 'table_prefix' cannot contain underscores."
                " Received a value of '%s'." % table_prefix
            )

        max_bins = 10 if max_bins is None else int(max_bins)
        if int(max_bins) < 2:
            raise ValueError(
                "Parameter 'max_bins' must be at least 2. Received a value of '%s'."
                % max_bins
            )

        if not project and not models:
            raise ValueError(
                "No project or model specified for performance definition creation.\n"
                "Please specify at least one of the two values."
            )

        # If no models were specified, search the supplied project for all models
        if not models:
            project = mr.get_project(project)
            models = mr.list_models(
                filter="eq(projectName, '{projectName}')".format(
                    projectName=project.name
                )
            )

        # Separate single models from multiple models
        if not isinstance(models, list):
            models = mr.get_model(models)
        else:
            # Collect all models into a list. This converts the PagedList response from mr.list_models to a normal list.
            for i, model in enumerate(models):
                models[i] = mr.get_model(model)
        if not project:
            project = mr.get_project(models[0].projectId)

        # Performance data cannot be captured unless certain project properties have been configured.
        for required in ["targetVariable", "targetLevel"]:
            if getattr(project, required, None) is None:
                raise ValueError(
                    "Project %s must have the '%s' property set."
                    % (project.name, required)
                )
        if (
            project.get("function") == "classification"
            and project.get("eventProbabilityVariable") is None
        ):
            raise ValueError(
                "Project %s must have the "
                "'eventProbabilityVariable' property set." % project.name
            )
        if (
            project.get("function") == "prediction"
            and project.get("predictionVariable") is None
        ):
            raise ValueError(
                "Project %s must have the 'predictionVariable' "
                "property set." % project.name
            )

        request = {
            "projectId": project.id,
            "name": name or project.name + " Performance",
            "modelIds": [model.id for model in models],
            "championMonitored": monitor_champion,
            "challengerMonitored": monitor_challenger,
            "maxBins": max_bins,
            "resultLibrary": output_library or "ModelPerformanceData",
            "includeAllData": all_data,
            "scoreExecutionRequired": scoring_required,
            "performanceResultSaved": save_output,
            "loadPerformanceResult": autoload_output,
            "dataLibrary": library_name or "Public",
            "description": description
            or "Performance definition for model "
            + ", ".join([model.name for model in models]),
            "casServerId": cas_server or "cas-shared-default",
            "dataPrefix": table_prefix,
            "traceOn": trace,
        }

        # If model doesn't specify input/output variables, try to pull from project definition
        if models[0].get("inputVariables", []):
            request["inputVariables"] = [
                v.get("name") for v in models[0]["inputVariables"]
            ]
            request["outputVariables"] = [
                v.get("name") for v in models[0]["outputVariables"]
            ]
        else:
            request["inputVariables"] = [
                v.get("name")
                for v in project.get("variables", [])
                if v.get("role") == "input"
            ]
            request["outputVariables"] = [
                v.get("name")
                for v in project.get("variables", [])
                if v.get("role") == "output"
            ]

        return cls.post(
            "/performanceTasks",
            json=request,
            headers={
                "Content-Type": "application/vnd.sas.models.performance.task+json"
            },
        )

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

        return cls.post("/performanceTasks/%s" % definition.id)

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

        return cls.get(
            "/workflowProcesses?filter=eq(associations.solutionObjectId,%22"
            + project["id"]
            + "%22)"
        )

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

        input = {
            "processName": workflow["name"],
            "processId": workflow["id"],
            "objectType": "MM_Project",
            "solutionObjectName": project_name,
            "solutionObjectId": project["id"],
            "solutionObjectUri": "/modelRepository/projects/" + project["id"],
            "solutionObjectMediaType": "application/vnd.sas.models.project+json",
        }

        # Note, you can get a HTTP Error 404: {"errorCode":74052,"message":"The workflow process for id <> cannot be found.
        #                                    Associations can only be made to running processes.","details":["correlator:
        #                                    e62c5562-2b11-45db-bcb7-933200cb0f0a","traceId: 3118c0fb1eb9702d","path:
        #                                    /modelManagement/workflowAssociations"],"links":[],"version":2,"httpStatusCode":404}
        # Which is fine and expected like the Visual Experience.
        return cls.post(
            "/workflowAssociations",
            json=input,
            headers={
                "Content-Type": "application/vnd.sas.workflow.object.association+json"
            },
        )

    @classmethod
    def create_custom_kpi(
        cls, model, project, timeLabel, kpiName, kpiValue, timeSK=None
    ):
        """Post a user supplied custom KPI to a SAS Model Manager project's MM_STD_KPI
        table. A custom KPI consists of the time label, KPI name, KPI value, and optionally
        the timeSK. Additionally, the model and project associated with the custom KPI are
        required.

        Multiple custom KPIs can be uploaded at once by passing lists in for the four
        arguments mentioned above.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the model.
        project : str or dict
        The name or id of the project, or a dictionary representation of
        the project.
        timeLabel : str or list
            Label associated with the dataset used within the performance definition.
        kpiName : str or list
            Name of the custom KPI.
        kpiValue : int or float or list
            Value of the custom KPI.
        timeSK : int or list, by default None
            Indicator for the MM_STD_KPI table to denote performance task order.
        """
        from .model_repository import ModelRepository

        mr = ModelRepository()

        # Step through options to determine project UUID
        if cls.is_uuid(project):
            projectId = project
        elif isinstance(project, dict) and "id" in project:
            projectId = project["id"]
        else:
            project = mr.get_project(project)
            projectId = project["id"]

        # Step through options to determine model UUID
        if cls.is_uuid(model):
            modelId = model
        elif isinstance(model, dict) and "id" in model:
            modelId = model["id"]
        else:
            model = mr.list_models(
                filter="eq('projectId','{}')&eq('name','{}')".format(projectId, model)
            )
            modelId = model["id"]

        model = mr.get_model(modelId)

        # If no timeSK is provided, create a list of 0's for the API call
        if not timeSK:
            timeSK = [0] * len(timeLabel)

        # Create a list of dicts mapped to each custom KPI value
        customKPI = [
            {"TimeLabel": label, "KPI": name, "Value": str(value), "TimeSK": SK}
            for label, name, value, SK in zip(timeLabel, kpiName, kpiValue, timeSK)
        ]
        headers = {"Accept": "application/vnd.sas.collection+json"}
        requestData = {"ProjectID": projectId, "ModelID": modelId, "KPIs": customKPI}
        # Include a terminal output, since it can take up to 60 seconds to POST the API
        print("Uploading custom kpis to SAS Viya...")
        return cls.post(
            "/projects/{}/kpis".format(projectId),
            headers=headers,
            data=json.dumps(requestData),
        )
