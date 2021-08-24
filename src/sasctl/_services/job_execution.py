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

        # TODO: parameters
        # TODO: definition id not RestObj passed
        # TODO: get link fails
        uri = cls.get_link(definition, 'self')['uri']

        data = {
            'jobDefinitionUri': uri
        }

        headers = {
            'Accept': 'application/vnd.sas.job.execution.job+json'
        }
        return cls.post('/jobs', headers=headers, json=data)