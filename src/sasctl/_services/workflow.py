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
    _SERVICE_ROOT= '/workflow'


    def list_workflow_enableddefinitions(self):
        """List all enabled Workflow Processes Definitions.

        Returns
        -------
        RestObj
            The list of workflows

        """

        #Additional header to fix 400 ERROR Bad Request
        headers={"Accept-Language": "en-US"}
        return self.get('/enabledDefinitions', headers=headers)
        
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
        
        ret = self.__find_specific_workflow(name)
        if ret is None:
            raise ValueError("No Workflow enabled for %s name or id." % name)
        
        if 'prompts' in ret:
            return ret['prompts']
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
        
        workflow = self.__find_specific_workflow(name)
        if workflow is None:
            raise ValueError("No Workflow enabled for %s name or id." % name)
        
        if input is None:
            return self.post('/processes?definitionId=' + workflow['id'],
	                     headers={'Content-Type': 'application/vnd.sas.workflow.variables+json'})
        if isinstance(input, dict):
            return self.post('/processes?definitionId=' + workflow['id'], json=input,
	                     headers={'Content-Type': 'application/vnd.sas.workflow.variables+json'})

    def __find_specific_workflow(self, name):
        # Internal helper method
        # Finds a workflow with the name (can be a name or id)
        # Returns a dict objects of the workflow
        listendef = self.list_workflow_enableddefinitions()
        for tmp in listendef:
            if tmp['name'] == name:
                return tmp
            elif tmp['id'] == name:
                return tmp
        
        # Did not find any enabled workflow with name/id
        return None


