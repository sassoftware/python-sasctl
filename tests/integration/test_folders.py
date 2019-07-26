#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import RestObj
from sasctl.services import folders

pytestmark = pytest.mark.usefixtures('session')

FOLDER_NAME = 'Test Folder'

def test_list_folders():
    all_folders = folders.list_folders()
    assert all(isinstance(f, RestObj) for f in all_folders)


def test_get_folder():
    folder = folders.get_folder('Resources')
    assert isinstance(folder, RestObj)
    assert 'Resources' == folder.name


def test_create_folder():
    folder = folders.create_folder(FOLDER_NAME)
    assert isinstance(folder, RestObj)
    assert FOLDER_NAME == folder.name


def test_delete_folder():
    folders.delete_folder(FOLDER_NAME)
    folder = folders.get_folder(FOLDER_NAME)
    assert folder is None
