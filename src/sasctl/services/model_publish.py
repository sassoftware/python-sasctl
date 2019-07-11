#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from sasctl.core import get, post

ROOT_PATH = '/modelPublish'


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

    name = re.sub(r'[^\w]', '', str(name))  # Remove all non-word characters

    if re.match(r'\d', name):               # Add a leading underscore if name starts with a digit
        name = '_' + name

    return name


def list_models():
    return get(ROOT_PATH + '/models').get('items', [])


def list_destinations():
    return get(ROOT_PATH + '/destinations').get('items', [])


def publish_model(model, destination, name=None, code=None, notes=None):
    from .model_repository import get_model, get_model_link

    code_types = {
        'ds2package': 'ds2',
        'datastep': 'datastep',
        '': ''
    }

    model = get_model(model)
    model_uri = get_model_link(model, 'self')

    # Get score code from registry if no code specified
    if code is None:
        code_link = get_model_link(model, 'scoreCode', True)
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
    return post(ROOT_PATH + '/models', json=request, headers={'Content-Type': 'application/vnd.sas.models.publishing.request+json'})


