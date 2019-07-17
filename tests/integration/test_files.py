#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import RestObj
from sasctl.services import files

pytestmark = pytest.mark.usefixtures('session')

FILENAME = 'dummy_file'

@pytest.fixture
def dummy_file(tmpdir):
    path = tmpdir.join(FILENAME + '.txt')
    path.write('Some test file content.')
    return str(path)


def test_list_files():
    all_files = files.list_files()

    assert all(isinstance(f, RestObj) for f in all_files)


@pytest.mark.incremental
class TestFile:
    filename = 'sasctl_test_file'

    def test_create_file_with_name(self, dummy_file):
        """Create a file with an explicitly set filename."""
        file = files.create_file(dummy_file, filename=self.filename)

        assert isinstance(file, RestObj)

    def test_get_file_with_name(self):
        """Ensure previously created file can be retrieved."""
        file = files.get_file(self.filename)

        assert isinstance(file, RestObj)
        assert self.filename == file.name

    def test_get_file_content(self, dummy_file):

        with open(dummy_file, 'r') as f:
            target = f.read()

        content = files.get_file_content(self.filename)

        assert target == content

    def test_delete_file_with_name(self):
        """Delete previously created file."""

        files.delete_file(self.filename)
        file = files.get_file(self.filename)

        assert file is None

    def test_create_file_without_name(self, dummy_file):
        """Create a file from just a path."""
        file = files.create_file(dummy_file)

        assert isinstance(file, RestObj)
        assert FILENAME == file.name

    def test_get_file_without_name(self):
        """Ensure previously created file can be retrieved."""
        file = files.get_file(FILENAME)

        assert isinstance(file, RestObj)
        assert FILENAME == file.name

    def test_delete_file_without_name(self):
        """Delete previously created file."""
        files.delete_file(FILENAME)
        file = files.get_file(FILENAME)

        assert file is None

    def test_create_file_from_file_object(self, dummy_file):
        """Create a file from just a path."""

        with open(dummy_file) as f:
            # Filename must be provided
            with pytest.raises(ValueError):
                file = files.create_file(f)

            file = files.create_file(f, filename=self.filename)

        assert isinstance(file, RestObj)
        assert self.filename == file.name

    def test_get_file_from_file_object(self):
        """Ensure previously created file can be retrieved."""
        file = files.get_file(self.filename)

        assert isinstance(file, RestObj)
        assert self.filename == file.name

    def test_delete_file_without_name(self):
        """Delete previously created file."""
        files.delete_file(self.filename)
        file = files.get_file(self.filename)

        assert file is None