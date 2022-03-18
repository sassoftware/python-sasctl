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
        """Define a new job that can be run in the SAS environment

        Parameters
        ----------
        name : str
            Job name
        description : str
            Job description
        type_ : {'casl', 'Compute'}
            Indicates type of code specified by `code`.  Use 'casl' if `code` is CASL or 'Compute' if `code` is
            data step.
        code : str
            Code to be executed whenever this job is run.
        parameters : list of dict
            List of parameters used by the job.  Each entry in the list should be a dictionary with the following items:
             - name : str
                parameter name
             - type : {'character', 'date', 'numeric', 'table'}
                parameter type
             - label : str
                human-readable label for parameter
             - required : bool
                is it required to specify a parameter value when running the job
             - defaultValue : any
                parameter's default value if not specified when running job
        properties : dict
            An arbitrary collection of name/value pairs to associate with the job definition.

        Returns
        -------
        RestObj

        """

        # Convert name:value pairs of properties to separate "name" & "value" fields.
        properties = properties or {}
        properties = [{'name': k, 'value': v} for k, v in properties.items()]

        parameters = parameters or []

        clean_parameters = []
        for param in parameters:
            param_type = str(param.get('type', '')).upper()
            if param_type not in ('TABLE', 'NUMERIC', 'DATE', 'CHARACTER'):
                raise ValueError(
                    "Type '{}' for parameter '{}' is invalid.  "
                    "Expected one of ('TABLE', 'NUMERIC', 'DATE', 'CHARACTER')".format(
                        param_type, param['name']
                    )
                )

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
