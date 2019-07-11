#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from sasctl.core import _build_crud_funcs, post


ROOT_PATH = '/projects'

list_projects, get_project, update_project, delete_project = _build_crud_funcs(ROOT_PATH + '/projects', 'project')

def create_project(name, description=None, image=None):
    """

    Parameters
    ----------
    name : str
    description : str
    image : str
        URI of an image to use as the project avatar

    Returns
    -------
    RestObj

    """

    body = {'name': name,
            'description': description,
            'imageUri': image
            }

    return post(ROOT_PATH + '/projects',
                json=body,
                headers={'Content-Type': 'application/vnd.sas.project+json'})