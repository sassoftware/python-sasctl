#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service


class Projects(Service):
    _SERVICE_ROOT = '/projects'

    list_projects, get_project, update_project, \
        delete_project = Service._crud_funcs('/projects', 'project')

    def create_project(self, name, description=None, image=None):
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

        return self.post('/projects', json=body,
                         headers={'Content-Type': 'application/vnd.sas.project+json'})