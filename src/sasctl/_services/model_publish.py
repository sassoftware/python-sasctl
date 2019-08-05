#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enables publishing objects such as models to various destinations."""

import re

from .service import Service
from .model_repository import ModelRepository


class ModelPublish(Service):
    """Enables publishing objects such as models to various destinations.

    The service provides methods for managing publish destinations like CAS,
    Hadoop, Teradata, or SAS Micro Analytic Service
    """

    _SERVICE_ROOT = '/modelPublish'
    _model_repository = ModelRepository()

    @staticmethod
    def _publish_name(name):
        """Create a valid module name from an input string.

        Model Publishing only permits names that adhere to the following
        restrictions:
            - Name must start with a letter or an underscore.
            - Name may not contain any spaces or special characters other than
              underscore.

        Parameters
        ----------
        name : str

        Returns
        -------
        str

        """
        # Remove all non-word characters
        name = re.sub(r'[^\w]', '', str(name))

        # Add a leading underscore if name starts with a digit
        if re.match(r'\d', name):
            name = '_' + name

        return name

    @classmethod
    def list_models(cls):
        return cls.get('/models').get('items', [])

    list_destinations, get_destination, update_destination, \
        delete_destination = Service._crud_funcs('/destinations',
                                                 'destination')

    @classmethod
    def publish_model(cls, model, destination, name=None, code=None,
                      notes=None):
        """

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model.
        destination : str or dict
            The name or id of the publishing destination, or a dictionary
            representation of the destination
        name : str, optional
            Name of the published model.  Defaults to the model name.
        code : str, optional
            The code to be published.
        notes

        Returns
        -------

        """
        code_types = {
            'ds2package': 'ds2',
            'datastep': 'datastep',
            '': ''
        }

        model = cls._model_repository.get_model(model)
        model_uri = cls._model_repository.get_model_link(model, 'cls')

        # Get score code from registry if no code specified
        if code is None:
            code_link = cls._model_repository.get_model_link(model,
                                                             'scoreCode',
                                                             True)
            if code_link:
                code = cls.get(code_link['href'])

        request = dict(
            name=name or model.get('name'),
            notes=notes,
            destinationName=destination,
        )

        modelContents = {
            'modelName': model.get('name'),
            'modelId': model.get('id'),
            'sourceUri': model_uri.get('href'),
            'publishLevel': 'model',        # ?? What are the options?
            'codeType': code_types[model.get('scoreCodeType', '').lower()],
            'codeUri': '',          # ??  Not needed if code is specified?
            'code': code
        }

        request['modelContents'] = [modelContents]
        return cls.post('/models', json=request, headers={
            'Content-Type':
                'application/vnd.sas.models.publishing.request+json'})
