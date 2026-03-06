#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock
import pytest
import tempfile
from pathlib import Path

from sasctl.pzmm import CodeFile


class TestValidateCodeFormatViaAPI:
    """Tests for _validate_code_format_via_api method."""

    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    def test_validate_code_format_success(self, mock_post):
        """Test successful code validation via API."""
        mock_post.return_value = {"valid": True}

        code = """
def execute():
    'Output:result'
    'DependentPackages:'
    result = 'test'
    return result
"""
        # Should not raise any exception
        CodeFile._validate_code_format_via_api(code)

        mock_post.assert_called_once_with(
            "/commons/validations/codeFiles",
            json={"content": code, "type": "decisionPythonFile"},
        )

    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    def test_validate_code_format_with_error_message(self, mock_post):
        """Test validation failure with error message."""
        mock_post.return_value = {
            "valid": False,
            "error": {
                "message": "Output docstring must be the first line in execute function"
            },
        }

        code = """
def execute():
    result = 'test'
    'Output:result'
    return result
"""
        with pytest.raises(ValueError, match="Output docstring must be the first line"):
            CodeFile._validate_code_format_via_api(code)

    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    def test_validate_code_format_with_error_no_message(self, mock_post):
        """Test validation failure with error but no message."""
        mock_post.return_value = {"valid": False, "error": "Validation failed"}

        code = "invalid code"

        with pytest.raises(ValueError, match="Validation failed"):
            CodeFile._validate_code_format_via_api(code)

    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    def test_validate_code_format_api_exception(self, mock_post):
        """Test handling of API exceptions during validation."""
        mock_post.side_effect = RuntimeError("API connection failed")

        code = "def execute():\n    return 1"

        with pytest.raises(
            ValueError, match="Code validation failed: API connection failed"
        ):
            CodeFile._validate_code_format_via_api(code)


class TestFindFileInFolder:
    """Tests for _find_file_in_folder method."""

    @mock.patch("sasctl.services.folders.get")
    def test_find_file_in_folder_found(self, mock_get):
        """Test finding an existing file in a folder."""
        mock_get.return_value = {
            "uri": "files/files/acde070d-8c4c-4f0d-9d8a-162843c10333"
        }

        result = CodeFile._find_file_in_folder("folder-456", "test.py")

        assert result is not None
        assert result == mock_get.return_value
        mock_get.assert_called_once_with(
            "/folders/folder-456/members",
            params={"filter": "and(eq(name, 'test.py'), eq(contentType, 'file'))"},
        )

    @mock.patch("sasctl.services.folders.get")
    def test_find_file_in_folder_not_found(self, mock_get):
        """Test when file is not found in folder."""
        mock_response = mock.MagicMock()
        mock_response.__len__ = mock.MagicMock(return_value=0)
        mock_get.return_value = mock_response

        result = CodeFile._find_file_in_folder("folder-456", "nonexistent.py")

        assert result is None

    @mock.patch("sasctl.services.folders.get")
    def test_find_file_in_folder_no_uri(self, mock_get):
        """Test when response has no URI."""
        mock_get.return_value = {"id": "unique-id"}

        result = CodeFile._find_file_in_folder("folder-456", "test.py")

        assert result is None


class TestLoadPythonCode:
    """Tests for _load_python_code method."""

    def test_load_python_code_from_string(self):
        """Test loading code from a string."""
        code = "def execute():\n    return 'test'"
        result = CodeFile._load_python_code(code)
        assert result == code

    def test_load_python_code_from_file(self):
        """Test loading code from a file path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def execute():\n    return 'test'")
            temp_path = Path(f.name)

        try:
            result = CodeFile._load_python_code(temp_path)
            assert result == "def execute():\n    return 'test'"
        finally:
            temp_path.unlink()

    def test_load_python_code_from_string_path(self):
        """Test loading code from a string path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test():\n    pass")
            temp_path = f.name

        try:
            result = CodeFile._load_python_code(temp_path)
            assert result == "def test():\n    pass"
        finally:
            Path(temp_path).unlink()

    def test_load_python_code_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Code cannot be empty"):
            CodeFile._load_python_code("")

    def test_load_python_code_whitespace_only(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Code cannot be empty"):
            CodeFile._load_python_code("   \n\t  ")

    def test_load_python_code_file_not_found(self):
        """Test that non-existent file raises ValueError."""
        with pytest.raises(ValueError, match="Code file not found"):
            CodeFile._load_python_code(Path("/nonexistent/path/to/file.py"))

    def test_load_python_code_empty_file(self):
        """Test that empty file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Code cannot be empty"):
                CodeFile._load_python_code(temp_path)
        finally:
            temp_path.unlink()

    def test_load_python_code_whitespace_only_file(self):
        """Test that file with only whitespace raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("   \n\n\t   ")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Code cannot be empty"):
                CodeFile._load_python_code(temp_path)
        finally:
            temp_path.unlink()

    def test_load_python_code_invalid_path_string(self):
        """Test that invalid path string is treated as raw code."""
        # A string that looks like it could be a path but is actually invalid
        code = "/some/path/that/does/not/exist.py but is actually code"
        result = CodeFile._load_python_code(code)
        assert result == code


class TestWriteIDCodeFile:
    """Tests for write_id_code_file method."""

    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    @mock.patch("sasctl.services.files.create_file")
    @mock.patch("sasctl.services.folders.get_folder")
    @mock.patch("sasctl.pzmm.code_file.CodeFile._find_file_in_folder")
    def test_write_id_code_file_success(
        self, mock_find_file, mock_get_folder, mock_create_file, mock_post
    ):
        """Test successful upload of a code file to Viya."""
        mock_folder_obj = mock.MagicMock()
        mock_folder_obj.id = "folder-123"
        mock_get_folder.return_value = mock_folder_obj

        mock_find_file.return_value = None

        mock_file_obj = mock.MagicMock()
        mock_file_obj.id = "12345"
        mock_file_obj.name = "test_code.py"
        mock_create_file.return_value = mock_file_obj

        mock_code_file = mock.MagicMock()
        mock_code_file.name = "test_code.py"
        mock_code_file.id = "cf-12345"
        mock_post.return_value = mock_code_file

        code = """
def execute():
    'Output:result'
    'DependentPackages:'
    result = 'test'
    return result
"""

        result = CodeFile.write_id_code_file(
            code=code,
            file_name="test_code.py",
            folder="/Public/TestFolder",
            validate_code=False,
        )

        assert mock_create_file.called
        assert mock_post.called
        assert result.name == "test_code.py"

        # Verify post was called with correct data
        mock_post.assert_called_once_with(
            "/codeFiles",
            json={
                "name": "test_code.py",
                "fileUri": "/files/files/12345",
                "type": "decisionPythonFile",
            },
        )

    @mock.patch("sasctl.pzmm.code_file.CodeFile._validate_code_format_via_api")
    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    @mock.patch("sasctl.services.files.create_file")
    @mock.patch("sasctl.services.folders.get_folder")
    @mock.patch("sasctl.pzmm.code_file.CodeFile._find_file_in_folder")
    def test_write_id_code_file_with_validation(
        self,
        mock_find_file,
        mock_get_folder,
        mock_create_file,
        mock_post,
        mock_validate,
    ):
        """Test upload with validation enabled."""
        mock_folder_obj = mock.MagicMock()
        mock_folder_obj.id = "folder-123"
        mock_get_folder.return_value = mock_folder_obj
        mock_find_file.return_value = None

        mock_file_obj = mock.MagicMock()
        mock_file_obj.id = "12345"
        mock_create_file.return_value = mock_file_obj

        mock_code_file = mock.MagicMock()
        mock_post.return_value = mock_code_file

        code = "def execute():\n    return 'test'"

        result = CodeFile.write_id_code_file(
            code=code,
            file_name="test_code.py",
            folder="/Public/TestFolder",
            validate_code=True,
        )

        # Verify validation was called
        mock_validate.assert_called_once_with(code)
        assert result == mock_code_file

    def test_write_id_code_file_invalid_filename(self):
        """Test that invalid file names are rejected."""
        code = """
def execute():
    'Output:result'
    'DependentPackages:'
    result = 42
"""

        with pytest.raises(ValueError, match="file_name must end with .py"):
            CodeFile.write_id_code_file(
                code=code, file_name="test_code.txt", folder="/Public/TestFolder"
            )

    @mock.patch("sasctl.services.folders.get_folder")
    @mock.patch("sasctl.pzmm.code_file.CodeFile._find_file_in_folder")
    def test_write_id_code_file_already_exists(self, mock_find_file, mock_get_folder):
        """Test that uploading a file that already exists raises error."""
        mock_folder_obj = mock.MagicMock()
        mock_folder_obj.id = "folder-123"
        mock_get_folder.return_value = mock_folder_obj

        mock_existing_file = mock.MagicMock()
        mock_existing_file.id = "existing-file-id"
        mock_existing_file.name = "duplicate.py"
        mock_find_file.return_value = mock_existing_file

        code = """
def execute():
    'Output:result'
    'DependentPackages:'
    result = 'test'
    return result
"""

        with pytest.raises(
            ValueError, match="File 'duplicate.py' already exists in this folder"
        ):
            CodeFile.write_id_code_file(
                code=code,
                file_name="duplicate.py",
                folder="/Public/TestFolder",
                validate_code=False,
            )

    @mock.patch("sasctl.services.folders.get_folder")
    def test_write_id_code_file_folder_not_found(self, mock_get_folder):
        """Test that referencing a non-existent folder raises error."""
        mock_get_folder.return_value = None

        code = """
def execute():
    'Output:result'
    'DependentPackages:'
    result = 'test'
    return result
"""

        with pytest.raises(ValueError, match="Folder '/NonExistent' not found"):
            CodeFile.write_id_code_file(
                code=code,
                file_name="test_code.py",
                folder="/NonExistent",
                validate_code=False,
            )

    def test_write_id_code_file_empty_code(self):
        """Test that empty code raises error."""
        with pytest.raises(ValueError, match="Code cannot be empty"):
            CodeFile.write_id_code_file(
                code="",
                file_name="test_code.py",
                folder="/Public/TestFolder",
                validate_code=False,
            )

    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    @mock.patch("sasctl.services.files.create_file")
    @mock.patch("sasctl.services.folders.get_folder")
    @mock.patch("sasctl.pzmm.code_file.CodeFile._find_file_in_folder")
    def test_write_id_code_file_from_path(
        self, mock_find_file, mock_get_folder, mock_create_file, mock_post
    ):
        """Test uploading code from a file path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def execute():\n    return 'test'")
            temp_path = Path(f.name)

        try:
            mock_folder_obj = mock.MagicMock()
            mock_folder_obj.id = "folder-123"
            mock_get_folder.return_value = mock_folder_obj
            mock_find_file.return_value = None

            mock_file_obj = mock.MagicMock()
            mock_file_obj.id = "12345"
            mock_create_file.return_value = mock_file_obj

            mock_code_file = mock.MagicMock()
            mock_post.return_value = mock_code_file

            result = CodeFile.write_id_code_file(
                code=temp_path,
                file_name="test_code.py",
                folder="/Public/TestFolder",
                validate_code=False,
            )

            assert result == mock_code_file
            mock_create_file.assert_called_once()
        finally:
            temp_path.unlink()

    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    @mock.patch("sasctl.services.files.delete_file")
    @mock.patch("sasctl.services.files.create_file")
    @mock.patch("sasctl.services.folders.get_folder")
    @mock.patch("sasctl.pzmm.code_file.CodeFile._find_file_in_folder")
    def test_write_id_code_file_post_fails_cleanup_success(
        self,
        mock_find_file,
        mock_get_folder,
        mock_create_file,
        mock_delete_file,
        mock_post,
    ):
        """Test that file is cleaned up when post fails."""
        mock_folder_obj = mock.MagicMock()
        mock_folder_obj.id = "folder-123"
        mock_get_folder.return_value = mock_folder_obj
        mock_find_file.return_value = None

        mock_file_obj = mock.MagicMock()
        mock_file_obj.id = "12345"
        mock_file_obj.__getitem__ = mock.MagicMock(return_value="12345")
        mock_create_file.return_value = mock_file_obj

        mock_post.side_effect = RuntimeError("API error")

        code = "def execute():\n    return 'test'"

        with pytest.raises(
            RuntimeError,
            match="There was an error with creating the code file: API error",
        ):
            CodeFile.write_id_code_file(
                code=code,
                file_name="test_code.py",
                folder="/Public/TestFolder",
                validate_code=False,
            )

        # Verify cleanup was attempted
        mock_delete_file.assert_called_once_with({"id": "12345"})

    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    @mock.patch("sasctl.services.files.delete_file")
    @mock.patch("sasctl.services.files.create_file")
    @mock.patch("sasctl.services.folders.get_folder")
    @mock.patch("sasctl.pzmm.code_file.CodeFile._find_file_in_folder")
    def test_write_id_code_file_post_fails_cleanup_fails(
        self,
        mock_find_file,
        mock_get_folder,
        mock_create_file,
        mock_delete_file,
        mock_post,
    ):
        """Test error handling when both post and cleanup fail."""
        mock_folder_obj = mock.MagicMock()
        mock_folder_obj.id = "folder-123"
        mock_get_folder.return_value = mock_folder_obj
        mock_find_file.return_value = None

        mock_file_obj = mock.MagicMock()
        mock_file_obj.id = "12345"
        mock_file_obj.__getitem__ = mock.MagicMock(return_value="12345")
        mock_create_file.return_value = mock_file_obj

        mock_post.side_effect = RuntimeError("API error")
        mock_delete_file.side_effect = RuntimeError("Delete failed")

        code = "def execute():\n    return 'test'"

        with pytest.raises(RuntimeError):
            CodeFile.write_id_code_file(
                code=code,
                file_name="test_code.py",
                folder="/Public/TestFolder",
                validate_code=False,
            )

    @mock.patch("sasctl.pzmm.code_file.CodeFile.post")
    @mock.patch("sasctl.services.files.create_file")
    @mock.patch("sasctl.services.folders.get_folder")
    @mock.patch("sasctl.pzmm.code_file.CodeFile._find_file_in_folder")
    def test_write_id_code_file_with_folder_object(
        self, mock_find_file, mock_get_folder, mock_create_file, mock_post
    ):
        """Test uploading with folder object instead of path."""
        mock_folder_obj = mock.MagicMock()
        mock_folder_obj.id = "folder-123"
        mock_get_folder.return_value = mock_folder_obj
        mock_find_file.return_value = None

        mock_file_obj = mock.MagicMock()
        mock_file_obj.id = "12345"
        mock_create_file.return_value = mock_file_obj

        mock_code_file = mock.MagicMock()
        mock_post.return_value = mock_code_file

        code = "def execute():\n    return 'test'"
        folder_dict = {"id": "folder-123", "name": "TestFolder"}

        result = CodeFile.write_id_code_file(
            code=code,
            file_name="test_code.py",
            folder=folder_dict,
            validate_code=False,
        )

        assert result == mock_code_file
        mock_get_folder.assert_called_once_with(folder_dict)
