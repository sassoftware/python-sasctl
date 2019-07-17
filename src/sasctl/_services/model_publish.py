#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from .service import Service


class ModelPublish(Service):
    """The Model Publish API provides support for publishing objects (such as
    models) to CAS, Hadoop, SAS Micro Analytic Service, or Teradata.
    """
    _SERVICE_ROOT = '/modelPublish'

    @staticmethod
    def _publish_name(name):
        """Create a valid module name from an input string.

        Model Publishing only permits names that adhere to the following restrictions:
            - Name must start with a letter or an underscore.
            - Name may not contain any spaces or special characters other than underscore.

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

    def list_models(self):
        return self.get('/models').get('items', [])

    def list_destinations(self):
        return self.get('/destinations').get('items', [])

    def publish_model(self, model, destination, name=None, code=None,
                      notes=None):

        code_types = {
            'ds2package': 'ds2',
            'datastep': 'datastep',
            '': ''
        }

        model = services.model_repository.get_model(model)
        model_uri = services.model_repository.get_model_link(model, 'self')

        # Get score code from registry if no code specified
        if code is None:
            code_link = services.model_repository.get_model_link(model,
                                                                 'scoreCode',
                                                                 True)
            if code_link:
                code = get(code_link['href'])

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
        return self.post('/models', json=request, headers={
            'Content-Type': 'application/vnd.sas.models.publishing.request+json'})


