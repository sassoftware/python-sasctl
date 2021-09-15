#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .service import Service
from .job_definitions import JobDefinitions


class JobExecution(Service):
    """The Job Execution service provides a common mechanism for executing and managing asynchronous jobs.

    This API works in conjunction with the Job Definitions service. You can use the Job Definitions service to create a
    definition for any existing Job Execution Provider. You can then execute this definition using the Job Execution
    service.
    """

    _SERVICE_ROOT = '/jobExecution'

    list_jobs, get_job, update_job, delete_job = Service._crud_funcs('/jobs')

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

        # Convert id/name into full definition object
        definition_obj = JobDefinitions.get_definition(definition)

        if definition_obj is None:
            raise ValueError("Unable to find job definition '%s'." % definition)

        uri = cls.get_link(definition_obj, 'self')['uri']

        parameters = parameters or {}

        data = {
            'name': name,
            'description': description,
            'jobDefinitionUri': uri,
            'arguments': parameters,
        }

        headers = {
            'Accept': 'application/vnd.sas.job.execution.job+json',
            'Content-Type': 'application/vnd.sas.job.execution.job.request+json',
        }
        return cls.post('/jobs', headers=headers, json=data)

    @classmethod
    def get_job_state(cls, job):
        """Check the status of an existing job.

        Parameters
        ----------
        job : str or dict
            The name or id of the job, or a dictionary representation of the job.

        Returns
        -------
        str
            {'pending', 'running', 'canceled', 'completed', failed'}

        """

        job_obj = cls.get_job(job)
        uri = '/jobs/%s/state' % job_obj.id

        return cls.get(uri)
