#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service


class Workflow(Service):
    """The Workflow API provides basic resources for list, prompt details,
    and running workflow processes.
    """

    _SERVICE_ROOT = "/workflow"

    def list_definitions(self, include_enabled=True, include_disabled=False):
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
        return self.get(url, headers={"Accept-Language": "en-US"})

    def list_enabled_definitions(self):
        """List process definitions that are currently enabled.

        Returns
        -------
        list
            The list of definitions.

        """
        return self.list_definitions(include_enabled=True, include_disabled=False)

    def list_workflow_prompt(self, name):
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

        ret = self._find_specific_workflow(name)
        if ret is None:
            raise ValueError("No Workflow enabled for %s name or id." % name)

        if "prompts" in ret:
            return ret["prompts"]
        else:
            # No prompt inputs on workflow
            return None

    def run_workflow_definition(self, name, input=None):
        """Runs specific Workflow Processes Definitions.

        Parameters
        ----------
        name : str
            Name or ID of an enabled workflow to execute
        input : dict, optional
            Input values for the workflow for initial workflow prompt

        Returns
        -------
        RestObj
            The executing workflow

        """

        workflow = self._find_specific_workflow(name)
        if workflow is None:
            raise ValueError("No Workflow enabled for %s name or id." % name)

        if input is None:
            return self.post(
                "/processes?definitionId=" + workflow["id"],
                headers={"Content-Type": "application/vnd.sas.workflow.variables+json"},
            )
        if isinstance(input, dict):
            return self.post(
                "/processes?definitionId=" + workflow["id"],
                json=input,
                headers={"Content-Type": "application/vnd.sas.workflow.variables+json"},
            )

    def _find_specific_workflow(self, name):
        # Internal helper method
        # Finds a workflow with the name (can be a name or id)
        # Returns a dict objects of the workflow
        listendef = self.list_enabled_definitions()
        for tmp in listendef:
            if tmp["name"] == name:
                return tmp
            elif tmp["id"] == name:
                return tmp

        # Did not find any enabled workflow with name/id
        return None
