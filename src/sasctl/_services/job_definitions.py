#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service


class JobDefinitions(Service):
    """
    Implements the Job Definitions REST API.

    The Job Definitions API manages jobs using the Create, Read, Update, and Delete operations.  A Job Definition
    is a batch execution that contains a list of input parameters, a job type, and a "code" attribute.  A job
    definition can run in multiple execution environments, based on the type of the job.

    See Also
    --------
    `REST Documentation <https://developer.sas.com/apis/rest/Compute/#job-definitions>`_
    """

    _SERVICE_ROOT = '/jobDefinitions'

    (
        list_definitions,
        get_definition,
        update_definition,
        delete_definition,
    ) = Service._crud_funcs('/definitions')

    # def get_definition(self, ):
    #     raise NotImplementedError()

    def get_summary(self):
        raise NotImplementedError()

    def get_headers(self):
        raise NotImplementedError()

    def create_definition(self):
        raise NotImplementedError()
