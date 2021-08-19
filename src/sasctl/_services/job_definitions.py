#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service
from ..core import RestObj


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

    @classmethod
    def create_definition(
        cls,
        name: str = None,
        description: str = None,
        type_: str = None,
        code: str = None,
        parameters=None,
        properties: dict = None,
    ) -> RestObj:
        """

        Parameters
        ----------
        name
        description
        type_
        code
        parameters
        properties

        Returns
        -------
        RestObj

        """

        # Convert name:value pairs of properties to separate "name" & "value" fields.
        properties = properties or {}
        properties = [{'name': k, 'value': v} for k, v in properties.items()]

        clean_parameters = []
        for param in parameters:
            param_type = str(param.get('type', '')).upper()
            if param_type not in ('TABLE', 'NUMERIC', 'DATE', 'CHARACTER'):
                # TODO: warn/raise
                continue

            new_param = {
                'version': 1,
                'name': str(param.get('name', ''))[
                    :100
                ],  # Max length of 100 characters
                'type': param_type,
                'label': str(param.get('label', ''))[
                    :250
                ],  # Max length of 250 characters
                'required': bool(param.get('required')),
                'defaultValue': param.get('defaultValue'),
            }
            clean_parameters.append(new_param)

        definition = {
            'version': 2,
            'name': name,
            'description': description,
            'type': type_,
            'code': code,
            'parameters': clean_parameters,
            'properties': properties,
        }

        return cls.post(
            '/definitions',
            json=definition,
            headers={'Accept': 'application/vnd.sas.job.definition+json'},
        )
