#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from sasctl.services import folders


def test_create_folder_basic():
    """Create a folder with minimal parameters."""
    FOLDER_NAME = "Spam"

    with mock.patch("sasctl._services.folders.Folders.post") as post:
        folders.create_folder(FOLDER_NAME)

    assert post.called
    json = post.call_args[1]["json"]
    params = post.call_args[1]["params"]
    assert json["name"] == FOLDER_NAME
    assert json["description"] is None
    assert params["parentFolderUri"] is None


def test_create_folder_with_desc():
    """Ensure description parameter is passed."""
    FOLDER_NAME = "Spam"
    FOLDER_DESC = "Created by sasctl testing."

    with mock.patch("sasctl._services.folders.Folders.post") as post:
        folders.create_folder(FOLDER_NAME, description=FOLDER_DESC)

    assert post.called
    json = post.call_args[1]["json"]
    params = post.call_args[1]["params"]
    assert json["name"] == FOLDER_NAME
    assert json["description"] == FOLDER_DESC
    assert params["parentFolderUri"] is None


def test_create_folder_with_parent():
    """Ensure parent folder parameter is handled correctly."""
    from sasctl.core import RestObj

    FOLDER_NAME = "Spam"

    # Mock response when retrieving parent folder.
    PARENT_FOLDER = RestObj(
        {
            "name": "",
            "id": "123",
            "links": [{"rel": "self", "uri": "/folders/somewhere/spam-eggs-spam-spam"}],
        }
    )

    with mock.patch(
        "sasctl._services.folders.Folders.get_folder", return_value=PARENT_FOLDER
    ):
        with mock.patch("sasctl._services.folders.Folders.post") as post:
            folders.create_folder(FOLDER_NAME, parent="Doesnt Matter")

    # Should have tried to create folder with correct name and parent URI
    assert post.called
    json = post.call_args[1]["json"]
    params = post.call_args[1]["params"]
    assert json["name"] == FOLDER_NAME
    assert json["description"] is None
    assert params["parentFolderUri"] == PARENT_FOLDER["links"][0]["uri"]

    # If parent folder can't be found, error should be raised
    with mock.patch("sasctl._services.folders.Folders.get_folder", return_value=None):
        with mock.patch("sasctl._services.folders.Folders.post"):
            with pytest.raises(ValueError):
                folders.create_folder(FOLDER_NAME, parent="Doesnt Matter")


@mock.patch("sasctl.core.Session.request")
def test_get_folder_by_name(request):
    """Verify that looking up a folder by name works."""
    request.return_value.status_code = 200
    request.return_value.json.return_value = {"name": "Spam"}

    folder = folders.get_folder("Spam")
    assert folder is not None
    assert folder.name == "Spam"

    # Service should do a search on folder name first followed by a get for folder details.
    assert request.call_count == 2

    url, params = request.call_args_list[0]
    search_filter = params["params"]

    # / should have been stripped out of folder name
    assert '"Spam"' in search_filter


@mock.patch("sasctl.core.Session.request")
def test_get_folder_by_path(request):
    """Verify that simple folder paths instead of names are handled."""
    request.return_value.status_code = 200
    request.return_value.json.return_value = {"name": "Spam"}

    folder = folders.get_folder("/Spam")
    assert folder is not None
    assert folder.name == "Spam"

    # Should not have called list_folders to search.  Should have been a direct call to lookup by path.
    assert request.call_count == 1

    url, params = request.call_args_list[0]
    query_string = params["params"]

    assert url[1] == "/folders/folders/@item"
    assert query_string["path"] == "/Spam"


@mock.patch("sasctl.core.Session.request")
def test_get_folder_by_path(request):
    """Verify that complex folder paths are handled."""
    request.return_value.status_code = 200
    request.return_value.json.return_value = {"name": "Spam"}

    folder = folders.get_folder("Spam/Eggs/Spam")
    assert folder is not None
    assert folder.name == "Spam"

    # Should not have called list_folders to search.  Should have been a direct call to lookup by path.
    assert request.call_count == 1

    url, params = request.call_args_list[0]
    query_string = params["params"]

    assert url[1] == "/folders/folders/@item"

    # Leading "/" should have been added
    assert query_string["path"] == "/Spam/Eggs/Spam"


@mock.patch("sasctl.core.Session.request")
def test_get_folder_by_delegate(request):
    """Verify that folder can be found by using delegate strings (e.g. @public)."""
    request.return_value.status_code = 200
    request.return_value.json.return_value = {"name": "Spam"}

    folder = folders.get_folder("@spam")
    assert folder is not None
    assert folder.name == "Spam"

    # Should not have called list_folders to search.  Should have been a direct call to GET by id.
    assert request.call_count == 1

    url, params = request.call_args_list[0]
    assert url[1] == "/folders/folders/@spam"
    assert params == {}
