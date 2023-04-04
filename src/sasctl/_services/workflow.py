#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime

from ..core import PagedItemIterator
from .service import Service


class Workflow(Service):
    """The Workflow API provides basic resources for list, prompt details,
    and running workflow processes.

    Warnings
    --------
    Note that this service is intentionally not included in the sasctl documentation.
    As of 2022.09 this service is not publicly documented on developer.sas.com and is
    only partially implemented here in order to provide functionality for the Model
    Manager service.  All methods included in this service should be treated as
    experimental and are subject to change without notice.

    """

    _SERVICE_ROOT = "/workflow"

    @classmethod
    def list_definitions(cls, include_enabled=True, include_disabled=False):
        """List workflow definitions.

        Parameters
        ----------
        include_enabled : bool, optional
            Include enabled definitions in the results.  Defaults to True.
        include_disabled : bool, optional
            Include disabled definitions in the results.  Defaults to False.

        Returns
        -------
        list
            The list of definitions.

        """
        if include_enabled and include_disabled:
            url = "/definitions"
        elif include_enabled:
            url = "/enabledDefinitions"
        elif include_disabled:
            url = "/disabledDefinitions"
        else:
            return []

        # Header required to prevent 400 ERROR Bad Request
        results = cls.get(url, headers={"Accept-Language": "en-US"})

        if results is None:
            return []
        if isinstance(results, (list, PagedItemIterator)):
            return results
        return [results]

    @classmethod
    def list_enabled_definitions(cls):
        """List process definitions that are currently enabled.

        Returns
        -------
        list
            The list of definitions.

        """
        return cls.list_definitions(include_enabled=True, include_disabled=False)

    @classmethod
    def list_workflow_prompt(cls, name):
        """List prompt Workflow Processes Definitions.

        Parameters
        ----------
        name : str
            Name or ID of an enabled workflow to retrieve inputs

        Returns
        -------
        list
            The list of prompts for specific workflow

        """

        ret = cls._find_specific_workflow(name)
        if ret is None:
            raise ValueError("No Workflow enabled for %s name or id." % name)

        if "prompts" in ret:
            return ret["prompts"]
        else:
            # No prompt inputs on workflow
            return None

    @classmethod
    def run_workflow_definition(cls, name, prompts=None):
        """Runs specific Workflow Processes Definitions.

        Parameters
        ----------
        name : str
            Name or ID of an enabled workflow to execute
        prompts : dict, optional
            Input values to provide for the initial workflow prompts.  Should be
            specified as name:value pairs.

        Returns
        -------
        RestObj
            The executing workflow

        """
        workflow = cls._find_specific_workflow(name)
        if workflow is None:
            raise ValueError("No Workflow enabled for %s name or id." % name)

        if prompts is None:
            return cls.post(
                "/processes?definitionId=" + workflow["id"],
                headers={"Content-Type": "application/vnd.sas.workflow.variables+json"},
            )
        if isinstance(prompts, dict):

            variables = []

            # For each prompt defined in the workflow, check if a value was provided.
            for prompt in workflow.prompts:
                if prompt["variableName"] in prompts:
                    name = prompt["variableName"]
                    value = prompts[name]

                    if type(value) == datetime.datetime:
                        # NOTE: do not use isinstance() to compare types as
                        # datetime.date will also evaluate as True.

                        # Explicitly convert to local time zone if not set.
                        if value.tzinfo is None:
                            try:
                                value = value.astimezone()
                            except OSError:
                                # On Windows pre-1970 dates will cause an issue.
                                # See https://bugs.python.org/issue36759
                                pass

                        if value.tzinfo is None:
                            # Failed to convert to local time.  Have to just assume it's UTC.
                            # Example: 2023-01-25T13:49:40.726162Z
                            value = value.isoformat() + "Z"
                        else:
                            # Example: 2023-01-25T13:49:40.726162-05:00
                            value = value.isoformat()

                    variables.append(
                        {
                            "name": name,
                            "value": value,
                            "scope": "local",
                            "type": prompt["variableType"],
                            "version": prompt["version"],
                        }
                    )

            return cls.post(
                "/processes?definitionId=" + workflow["id"],
                json={"variables": variables},
                headers={"Content-Type": "application/vnd.sas.workflow.variables+json"},
            )

    @classmethod
    def _find_specific_workflow(cls, name):
        # Internal helper method
        # Finds a workflow with the name (can be a name or id)
        # Returns a dict objects of the workflow
        listendef = cls.list_enabled_definitions()
        for tmp in listendef:
            if tmp["name"] == name:
                return tmp
            elif tmp["id"] == name:
                return tmp

        # Did not find any enabled workflow with name/id
        return None
