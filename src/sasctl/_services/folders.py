#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from sasctl.utils.cli import sasctl_command

from .service import Service


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
                raise ValueError(f"`parent` folder {parent} does not exist.")
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
            May be one of:

            - folder name
            - folder ID
            - folder path
            - folder delegate string
            - dictionary representation of the folder
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

        Examples
        --------
        The following four examples are all functionally equivalent.

        >>> get_folder("Public")
        {"name": "Public", "id": "4a737209-5662"}

        >>> get_folder("/Public")
        {"name": "Public", "id": "4a737209-5662"}

        >>> get_folder("@public")
        {"name": "Public", "id": "4a737209-5662"}

        >>> get_folder("@public")
        {"name": "Public", "id": "4a737209-5662"}

        >>> get_folder("4a737209-5662")
        {"name": "Public", "id": "4a737209-5662"}

        The full folder path can also be specified.

        >>> get_folder("/Public/Demo")
        {"name": "Demo", "id": "148081bf-1c86"}

        Special folders can be identified using a delegate string.  Currently supported
        are: @myFolder, @appDataFolder, @myHistory, @myFavorites, and @public.

        >>> get_folder("@myFolder")
        {"name": "My Folder", "id": "71687cd2-db4b"}

        """
        # If a folder path is specified, lookup folder using the full path instead of the name.
        if isinstance(folder, str) and "/" in folder:
            # Path must include a leading "/"
            if not folder.startswith("/"):
                folder = f"/{folder}"

            return cls.get("/folders/@item", params={"path": folder})

        # Its possible to lookup special folders by using a handle (called a delegate string in docs)
        # Current values (2022.09) are @myFolder, @appDataFolder, @myHistory, @myFavorites, @public.
        if isinstance(folder, str) and folder.startswith("@"):
            return cls.get(f"/folders/{folder}")

        return cls._get_folder(folder, refresh=refresh)

    @classmethod
    def create_path(cls, folder, description=None):
        """Create a new folder recursively.

        Parameters
        ----------
        folder : str
            The folder to be created including the path.
        description: str, optional
             A description of the folder

        Returns
        -------
        RestObj
            Details of newly-created folder

        """
        folder = str(folder)

        # Path must include a leading "/"
        if not folder.startswith("/"):
            folder = f"/{folder}"
        path = folder.split("/")

        for level in range(2, len(path) + 1):
            current_path = path[0:level]
            name = current_path[-1]
            parent = "/".join(current_path[0:-1]) or None
            new_folder = cls.get_folder("/".join(current_path))
            if not new_folder:
                new_folder = cls.create_folder(
                    name, parent=parent, description=description
                )
        return new_folder
