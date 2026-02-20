# Copyright (c) 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tools for creating and uploading Python code files for SAS Intelligent Decisioning.
"""

# Standard Library Imports
from pathlib import Path
from typing import Union

# Package Imports
from ..core import RestObj
from ..services import files as file_service
from ..services import folders as folders_service
from .._services.service import Service

class CodeFile(Service):
    """
    A class for creating Python code files formatted for SAS Intelligent Decisioning.

    SAS Intelligent Decisioning requires Python code files to follow a specific format
    with an execute function that includes docstrings for output variables and
    dependent packages.
    """

    _SERVICE_ROOT = "/decisions"


    @classmethod
    def _validate_code_format_via_api(cls, code: str) -> bool:
        """
        Validate code format using the SAS Viya validation endpoint.

        This validates Output docstring position, return statements, execute function,
        and other ID-specific formatting requirements.

        Parameters
        ----------
        code : str
            Python code to validate.

        Raises
        ------
        ValueError
            If the code doesn't meet ID formatting requirements.
        """
        try:
            response = cls.post(
                "/commons/validations/codeFiles",
                json={"content": code, "type": "decisionPythonFile"}
            )

            # If validation fails, the response will contain an error
            if not response.get('valid', True):
                error = response.get('error', {})
                if isinstance(error, dict):
                    error_message = error.get('message', str(error))
                else:
                    error_message = str(error)
                raise ValueError(error_message)
                
        except Exception as e:
            # Re-raise ValueError as-is, wrap other exceptions
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Code validation failed: {str(e)}")

    @classmethod
    def _find_file_in_folder(cls, folder_id: str, file_name: str) -> Union[RestObj, None]:
        """
        Find a file in a specific folder by name.

        Parameters
        ----------
        folder_id : str
            The ID of the folder to search in.
        file_name : str
            Name of the file to find.

        Returns
        -------
        RestObj or None
            File details if found, None otherwise.
        """
        from ..services import folders as folders_service
        
        # Search for the file in the folder
        file_filter = f"and(eq(name, '{file_name}'), eq(contentType, 'file'))"
        response = folders_service.get(
            f"/folders/{folder_id}/members",
            params={"filter": file_filter}
        )
        
        if len(response) <= 0:
            # No files with file_name were found.
            return None
        
        file_uri = response.get('uri')
        
        if file_uri:
            return response
        
        return None
    
    @classmethod
    def _load_python_code(
        cls, code: Union[str, Path]
    ) -> str:
        """
        Load and prepare a Python code file for SAS Intelligent Decisioning.

        This method loads code from a string or file path and performs basic checks.
        Actual validation against ID format requirements happens during upload.

        Parameters
        ----------
        code : str or pathlib.Path
            Python code as a string or path to a Python file.

        Returns
        -------
        str
            The Python code file content.

        Raises
        ------
        ValueError
            If code is empty or file is not found.
        """
        # Check for empty string first
        if isinstance(code, str) and (not code or not code.strip()):
            raise ValueError("Code cannot be empty")

        # Convert string path to Path object if needed (with error handling for invalid paths)
        try:
            if isinstance(code, str) and Path(code).exists():
                code = Path(code)
        except OSError:
            # Path is invalid (e.g., too long or malformed) - treat as raw code string
            pass

        if isinstance(code, Path):
            if not code.exists():
                raise ValueError(f"Code file not found: {code}")
            code = code.read_text()

            if not code or not code.strip():
                raise ValueError("Code cannot be empty")

        return code

    @classmethod
    def write_id_code_file(
        cls,
        code: Union[str, Path],
        file_name: str,
        folder: Union[str, dict],
        validate_code: bool = True,
    ) -> RestObj:
        """
        Validate and upload a Python code file to SAS Intelligent Decisioning.

        This method validates a properly formatted ID Python code file and uploads
        it to a specified folder in SAS Viya, then registers it with the Decisions service.

        Parameters
        ----------
        code : str or pathlib.Path
            Python code as a string or path to a Python file. The code must already
            be formatted for ID with an execute function and proper docstrings.
        file_name : str
            Name for the code file (e.g., 'my_code.py'). Must end with .py
        folder : str or dict
            Target folder in SAS Viya. Can be a folder name, path (e.g.,
            '/Public/MyFolder'), or folder object returned by folders.get_folder().
        validate_code: bool
            If True, validates code format via API before upload. If False, skips validation.

        Returns
        -------
        RestObj
            Code file object returned by the Decisions service.

        Raises
        ------
        ValueError
            If file_name doesn't end with .py, if folder is not found, if code
            doesn't contain required docstrings, or if code is invalid.
        SyntaxError
            If the provided code has syntax errors.
        """
        # Validate file_name
        if not file_name.endswith(".py"):
            raise ValueError("file_name must end with .py extension")

        # Load the code (handles file paths, empty checks, etc.)
        loaded_code = cls._load_python_code(code)

        # Validate code format if requested
        if validate_code:
            cls._validate_code_format_via_api(loaded_code)

        # Verify that the folder exists
        folder_obj = folders_service.get_folder(folder)
        if not folder_obj:
            raise ValueError(f"Folder '{folder}' not found")
        
        # Verify that a file with that name doesn't exist
        file_obj = cls._find_file_in_folder(folder_obj.id, file_name)
        if file_obj:
            raise ValueError(f"File '{file_name}' already exists in this folder.")

        # Upload the file to Viya Files service
        file_obj = file_service.create_file(
            file=loaded_code.encode("utf-8"),
            folder=folder,
            filename=file_name,
        )

        data = {
            "name": file_name,
            "fileUri": f"/files/files/{file_obj.id}",
            "type": "decisionPythonFile",
        }

        try:
            code_file = cls.post("/codeFiles", json=data)
        except Exception as post_error:
            # Try to clean up the uploaded file since code file creation failed
            try:
                # There is no response from deleting a file object
                file_service.delete_file({"id": file_obj['id']})

            except Exception as delete_error:
                raise RuntimeError(
                    f"There was an error creating the code file: {post_error}. "
                    f"Additionally, failed to delete the orphaned file: {delete_error}"
                )
            raise RuntimeError(f"There was an error with creating the code file: {post_error}")


        return code_file
