#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl.core import RestObj
from sasctl.services import folders

pytestmark = pytest.mark.usefixtures("session")


@pytest.mark.incremental
class TestFolders:
    FOLDER_NAME = "Test Folder"

    def test_list_folders(self):
        all_folders = folders.list_folders()
        assert all(isinstance(f, RestObj) for f in all_folders)

    # def test_get_folder(self):
    #     folder = folders.get_folder('Resources')
    #     assert isinstance(folder, RestObj)
    #     assert 'Resources' == folder.name

    def test_create_folder(self):
        folder = folders.create_folder(self.FOLDER_NAME)
        assert isinstance(folder, RestObj)
        assert self.FOLDER_NAME == folder.name

    def test_get_folder(self):
        folder = folders.get_folder(self.FOLDER_NAME)
        assert isinstance(folder, RestObj)
        assert folder.name == self.FOLDER_NAME

    def test_list_with_pagination(self):
        some_folders = folders.list_folders(limit=2)

        # PagedList with technically include all results, so subset to just
        # the results that were explicitly requested
        some_folders = some_folders[:2]

        assert isinstance(some_folders, list)
        assert all(isinstance(f, RestObj) for f in some_folders)

        other_folders = folders.list_folders(start=2, limit=3)
        assert isinstance(other_folders, list)
        other_folders = other_folders[:3]

        assert all(f not in some_folders for f in other_folders)

    def test_delete_folder(self):
        folders.delete_folder(self.FOLDER_NAME)
        folder = folders.get_folder(self.FOLDER_NAME)
        assert folder is None
