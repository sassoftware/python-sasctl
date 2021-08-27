#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from unittest import mock

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
class TestTextFile:
    filename = 'sasctl_test_file'

    def test_create_file_with_name(self, dummy_file):
        # Requests uses os.urandom(16) to generate boundaries for multi-part
        # form uploads.  Mock the output to ensure a consistent value so
        # that body request/responses always match.
        with mock.patch('os.urandom', return_value='abcdefghijklmnop'.encode('utf-8')):

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
        # Requests uses os.urandom(16) to generate boundaries for multi-part
        # form uploads.  Mock the output to ensure a consistent value so
        # that body request/responses always match.
        with mock.patch('os.urandom', return_value='abcdefghijklmnop'.encode('utf-8')):
            file = files.create_file(dummy_file)

        assert isinstance(file, RestObj)
        assert FILENAME == file.name

    def test_get_file_without_name(self):  # skipcq: PYL-R0201
        """Ensure previously created file can be retrieved."""
        file = files.get_file(FILENAME)

        assert isinstance(file, RestObj)
        assert FILENAME == file.name

    def test_delete_file_without_name(self):  # skipcq: PYL-R0201
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

            # Requests uses os.urandom(16) to generate boundaries for multi-part
            # form uploads.  Mock the output to ensure a consistent value so
            # that body request/responses always match.
            with mock.patch(
                'os.urandom', return_value='abcdefghijklmnop'.encode('utf-8')
            ):
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


@pytest.mark.incremental
class TestPickleFile:
    filename = 'sasctl_test_pickle_file'

    def test_create_file_with_name(self, dummy_file):
        import io

        # Requests uses os.urandom(16) to generate boundaries for multi-part
        # form uploads.  Mock the output to ensure a consistent value so
        # that body request/responses always match.
        with mock.patch('os.urandom', return_value='abcdefghijklmnop'.encode('utf-8')):

            # Read the file contents and pickle
            with open(dummy_file, 'r') as f:
                file = io.BytesIO(pickle.dumps(f.read()))

            """Create a file with an explicitly set filename."""
            file = files.create_file(file, filename=self.filename)

        assert isinstance(file, RestObj)

    def test_get_file_with_name(self):
        """Ensure previously created file can be retrieved."""
        file = files.get_file(self.filename)

        assert isinstance(file, RestObj)
        assert self.filename == file.name

    def test_get_file_content(self, dummy_file):

        with open(dummy_file, 'r') as f:
            target = f.read()

        # Should return binary pickle of text file contents
        content = files.get_file_content(self.filename)

        # Betamax recording seems to add 4 extra bytes to the beginning?
        # Doesn't occur during real requests.
        try:
            result = pickle.loads(content)
        except (KeyError, pickle.UnpicklingError):
            result = pickle.loads(content[4:])

        assert target == result

    def test_delete_file_with_name(self):
        """Delete previously created file."""

        files.delete_file(self.filename)
        file = files.get_file(self.filename)

        assert file is None

    def test_create_file_without_name(self, dummy_file):  # skipcq: PYL-R0201
        """Create a file from just a path."""
        # Requests uses os.urandom(16) to generate boundaries for multi-part
        # form uploads.  Mock the output to ensure a consistent value so
        # that body request/responses always match.
        with mock.patch('os.urandom', return_value='abcdefghijklmnop'.encode('utf-8')):
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

            # Requests uses os.urandom(16) to generate boundaries for multi-part
            # form uploads.  Mock the output to ensure a consistent value so
            # that body request/responses always match.
            with mock.patch(
                'os.urandom', return_value='abcdefghijklmnop'.encode('utf-8')
            ):
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
