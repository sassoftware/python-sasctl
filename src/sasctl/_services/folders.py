#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service
from sasctl.utils.cli import sasctl_command


class Folders(Service):
    """The Folders API provides an organizational structure for SAS and
    external content. It can also be used for favorites folders or a history
    of objects accessed. The resources that are stored in folders (members)
    use a URI to point back to those resources.

    """
    _SERVICE_ROOT = '/folders'

    list_folders, get_folder, update_folder, \
    delete_folder = Service._crud_funcs('/folders', 'folder')

    @sasctl_command('folders', 'create')
    def create_folder(self, name, parent=None, description=None):
        """

        Parameters
        ----------
        name : str
            The name of the new folder
        parent : str or dict, optional
            The parent folder for this folder, if any.  Can be a folder name,
            id, or dict response from `get_folder`
        description : str, optional
            A description of the folder

        Returns
        -------

        """
        parent = self.get_folder(parent)

        body = {'name': name,
                'description': description,
                'folderType': 'folder',
                'parentFolderUri': parent.id if parent else None }

        return self.post('/folders',
                         json=body,
                         headers={'Content-Type': 'application/vnd.sas.content.folder+json'})