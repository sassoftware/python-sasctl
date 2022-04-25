#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from .folders import Folders
from .service import Service
from sasctl.utils.cli import sasctl_command


class Files(Service):
    """The Files API provides persistence and retrieval of files, such as
    documents, attachments, and reports.

    The file can be associated with the URI of another identifiable object
    (for example, a parentUri). Every file must have an assigned content type
    and name. Files can be retrieved individually by using the file's
    identifier or as a list of files by using a parentUri. Each file has its
    content stream associated with it.  After creation, the metadata that is
    associated with the file or the actual content can be updated. A single
    file can be deleted by using a specific ID. Multiple files can be deleted
    by specifying a parentUri.  A file can be uploaded via raw request or
    multipart form request.
    """

    _SERVICE_ROOT = "/files"

    list_files, get_file, update_file, delete_file = Service._crud_funcs(
        "/files", "file"
    )

    @classmethod
    @sasctl_command("files", "create")
    def create_file(cls, file, folder=None, filename=None, expiration=None):
        """Create a new file on the server by uploading a local file.

        Parameters
        ----------
        file : str or file_like
            Path to the file to upload or a file-like object.
        folder : str or dict, optional
            Name, or, or folder information as returned by :func:`.get_folder`.
        filename : str, optional
            Name to assign to the uploaded file.  Defaults to the filename if `file` is a path, otherwise required.
        expiration : datetime, optional
            A timestamp that indicates when to expire the file.  Defaults to no expiration.

        Returns
        -------
        RestObj
            A dictionary containing the file attributes.

        """
        if isinstance(file, str):
            filename = filename or os.path.splitext(os.path.split(file)[1])[0]

            with open(file, "rb") as f:
                file = f.read()
        else:
            if filename is None:
                raise ValueError(
                    "`filename` must be specified if `file` is not a path."
                )

            file = file.read()

        params = {}

        if folder is not None:
            _folder = Folders().get_folder(folder)

            if _folder is None:
                raise ValueError("Folder '%s' could not be found." % folder)

            params["parentFolderUri"] = cls.get_link(_folder, "self")["href"]

        if expiration is not None:
            pass
            # TODO: add 'expirationTimeStamp' to params.  Need to determine correct format

        return cls.post("/files#multipartUpload", files={filename: file}, params=params)

    @classmethod
    @sasctl_command("files", "content")
    def get_file_content(cls, file):
        """Download the contents of a file.

        Parameters
        ----------
        file : str or dict, optional
            Name or file information as returned by :func:`get_file`.

        Returns
        -------
        content

        """
        file = cls.get_file(file)

        r = cls.request_link(file, "content", format="response")

        content_type = r.headers.get("Content-Type", "")

        if "text/plain" in content_type:
            return r.text
        if "application/json" in content_type:
            return r.json()
        if "application/octet-stream" in content_type:
            return r.content

        return r.text
