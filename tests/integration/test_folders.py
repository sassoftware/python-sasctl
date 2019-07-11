#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import RestObj


pytestmark = pytest.mark.usefixtures('session')

FOLDER_NAME = 'Test Folder'

def test_list_folders():
    from sasctl.services.folders import list_folders

    folders = list_folders()
    assert all(isinstance(f, RestObj) for f in folders)


def test_get_folder():
    from sasctl.services.folders import get_folder

    folder = get_folder('Resources')
    assert isinstance(folder, RestObj)
    assert 'Resources' == folder.name


def test_create_folder():
    from sasctl.services.folders import create_folder

    folder = create_folder(FOLDER_NAME)
    assert isinstance(folder, RestObj)
    assert FOLDER_NAME == folder.name


def test_delete_folder():
    from sasctl.services.folders import delete_folder, get_folder

    delete_folder(FOLDER_NAME)
    folder = get_folder(FOLDER_NAME)
    assert folder is None
