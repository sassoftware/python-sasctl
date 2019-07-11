#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from sasctl.core import post, _build_crud_funcs
from sasctl.utils.cli import sasctl_command

_SERVICE_ROOT = '/folders'

list_folders, get_folder, update_folder, delete_folder = _build_crud_funcs(_SERVICE_ROOT + '/folders', 'folder')


@sasctl_command('folders', 'create')
def create_folder(name, parent=None, description=None):
    """

    Parameters
    ----------
    name : str
        The name of the new folder
    parent : str or dict, optional
        The parent folder for this folder, if any.  Can be a folder name, id, or dict response from get_folder
    description : str, optional
        A description of the folder

    Returns
    -------

    """
    parent = get_folder(parent)

    body = {'name': name,
            'description': description,
            'folderType': 'folder',
            'parentFolderUri': parent.id if parent else None }

    return post(_SERVICE_ROOT + '/folders',
                json=body,
                headers={'Content-Type': 'application/vnd.sas.content.folder+json'})