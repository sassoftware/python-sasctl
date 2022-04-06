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
    FOLDER_NAME = 'Spam'

    with mock.patch('sasctl._services.folders.Folders.post') as post:
        folders.create_folder(FOLDER_NAME)

    assert post.called
    json = post.call_args[1]['json']
    params = post.call_args[1]['params']
    assert json['name'] == FOLDER_NAME
    assert json['description'] is None
    assert params['parentFolderUri'] is None


def test_create_folder_with_desc():
    """Ensure description parameter is passed."""
    FOLDER_NAME = 'Spam'
    FOLDER_DESC = 'Created by sasctl testing.'

    with mock.patch('sasctl._services.folders.Folders.post') as post:
        folders.create_folder(FOLDER_NAME, description=FOLDER_DESC)

    assert post.called
    json = post.call_args[1]['json']
    params = post.call_args[1]['params']
    assert json['name'] == FOLDER_NAME
    assert json['description'] == FOLDER_DESC
    assert params['parentFolderUri'] is None


def test_create_folder_with_parent():
    """Ensure parent folder parameter is handled correctly."""
    from sasctl.core import RestObj

    FOLDER_NAME = 'Spam'

    # Mock response when retrieving parent folder.
    PARENT_FOLDER = RestObj({'name': '', 'id': '123', 'links': [
        {'rel': 'self', 'uri': '/folders/somewhere/spam-eggs-spam-spam'}
    ]})

    with mock.patch('sasctl._services.folders.Folders.get_folder', return_value=PARENT_FOLDER):
        with mock.patch('sasctl._services.folders.Folders.post') as post:
            folders.create_folder(FOLDER_NAME, parent='Doesnt Matter')

    # Should have tried to create folder with correct name and parent URI
    assert post.called
    json = post.call_args[1]['json']
    params = post.call_args[1]['params']
    assert json['name'] == FOLDER_NAME
    assert json['description'] is None
    assert params['parentFolderUri'] == PARENT_FOLDER['links'][0]['uri']

    # If parent folder can't be found, error should be raised
    with mock.patch('sasctl._services.folders.Folders.get_folder', return_value=None):
        with mock.patch('sasctl._services.folders.Folders.post'):
            with pytest.raises(ValueError):
                folders.create_folder(FOLDER_NAME, parent='Doesnt Matter')
