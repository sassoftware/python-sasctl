#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .service import Service


class JobExecution(Service):
    """The Job Execution service provides a common mechanism for executing and managing asynchronous jobs.

    This API works in conjunction with the Job Definitions service. You can use the Job Definitions service to create a
    definition for any existing Job Execution Provider. You can then execute this definition using the Job Execution
    service.
    """

    _SERVICE_ROOT = '/jobExecution'

    @classmethod
    def create_job(cls, definition, name=None, description=None, parameters=None):
        """Execute a job from an existing job definition.

        Parameters
        ----------
        definition : str or RestObj
        name : str, optional
            Name for the requested job.
        description : str, optional
            Description of the requested job
        parameters : dict, optional
            Parameter name/value pairs used to overwrite default parameters defined in job definition.

        Returns
        -------
        RestObj
            Job request details

        """

        # TODO: definition id not RestObj passed
        # TODO: get link fails
        uri = cls.get_link(definition, 'self')['uri']

        parameters = parameters or {}
        # TODO: expiresAfter - set to reasonable default

        data = {
            'name': name,
            'description': description,
            'jobDefinitionUri': uri,
            'arguments': parameters
        }

        headers = {
            'Accept': 'application/vnd.sas.job.execution.job+json',
            'Content-Type': 'application/vnd.sas.job.execution.job.request+json'
        }
        return cls.post('/jobs', headers=headers, json=data)
