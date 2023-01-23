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

    _SERVICE_ROOT = "/folders"

    list_folders, _get_folder, update_folder, delete_folder = Service._crud_funcs(
        "/folders", "folder"
    )

    @classmethod
    @sasctl_command("folders", "create")
    def create_folder(cls, name, parent=None, description=None):
        """Create a new folder.

        Parameters
        ----------
        name : str
            The name of the new folder
        parent : str or dict, optional
            The parent folder for this folder, if any.  Can be a folder name, id, or dict response from `get_folder`.
            If not specified, new folder will be created under root folder.
        description : str, optional
            A description of the folder

        Returns
        -------
        RestObj
            Details of newly-created folder

        """
        if parent is not None:
            parent_obj = cls.get_folder(parent)

            parent_uri = cls.get_link(parent_obj, "self")
            if parent_uri is None:
                raise ValueError("`parent` folder '%s' does not exist." % parent)
            parent_uri = parent_uri["uri"]
        else:
            parent_uri = None

        body = {"name": name, "description": description, "folderType": "folder"}

        return cls.post(
            "/folders",
            json=body,
            params={"parentFolderUri": parent_uri},
            headers={"Content-Type": "application/vnd.sas.content.folder+json"},
        )

    @classmethod
    def get_folder(cls, folder, refresh=False):
        """Return a folder instance.

        Parameters
        ----------
        folder : str or dict
            Name, ID, or dictionary representation of the folder.
        refresh : bool, optional
            Obtain an updated copy of the folder.

        Returns
        -------
        RestObj or None
            A dictionary containing the folder attributes or None.

        Notes
        -------
        If `folder` is a complete representation of the folder it will be
        returned unless `refresh` is set.  This prevents unnecessary REST
        calls when data is already available on the client.

        """
        # If users pass in a folder name like "/Public" instead of "Public"
        # then just cleanup the folder name so that it matches what's returned by Viya.
        if isinstance(folder, str):
            folder = folder.strip("/")

        return cls._get_folder(folder, refresh=refresh)
