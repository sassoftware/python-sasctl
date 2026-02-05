# Copyright (c) 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tools for creating and uploading Python code files for SAS Intelligent Decisioning.
"""

# Standard Library Imports
import ast
from pathlib import Path
from typing import List, Union

# Package Imports
from ..core import RestObj
from ..services import files as file_service
from ..utils.misc import IMPORT_TO_INSTALL_MAPPING
from .write_json_files import JSONFiles
from .._services.service import Service

class CodeFile(Service):
    """
    A class for creating Python code files formatted for SAS Intelligent Decisioning.
    
    SAS Intelligent Decisioning requires Python code files to follow a specific format
    with an execute function that includes docstrings for output variables and 
    dependent packages.
    """
    
    _SERVICE_ROOT = "/decisions"
    
    # Constants for required ID code file elements
    EXECUTE_FUNCTION_NAME = "execute"
    OUTPUT_DOCSTRING_PREFIX = "Output:"
    DEPENDENT_PACKAGES_DOCSTRING_PREFIX = "DependentPackages:"


    @classmethod
    def _auto_detect_dependencies(cls, code: str) -> List[str]:
        """
        Auto-detect package dependencies from Python code.
        
        Parameters
        ----------
        code : str
            Python code to analyze.
            
        Returns
        -------
        list of str
            List of detected package names.
        """
        # Parse the code to get imports from the abstract syntax tree
        try:
            tree = ast.parse(code)
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        modules.add(node.module.split(".")[0])
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        modules.add(name.name.split(".")[0])

            # Filter out standard library modules
            modules = list(modules)
            modules = JSONFiles.remove_standard_library_packages(modules)
            return sorted(modules)
        except Exception:
            return []

    @classmethod
    def _extract_docstring_variables(cls, code: str, docstring_prefix: str) -> List[str]:
        """
        Extract variables from a docstring line.
        
        Parameters
        ----------
        code : str
            Python code containing the docstring.
        docstring_prefix : str
            The prefix to search for (e.g., 'Output:' or 'DependentPackages:').
            
        Returns
        -------
        list of str
            List of variable/package names from the docstring.
            
        Raises
        ------
        ValueError
            If the docstring is not found.
        """
        matching_lines = [
            line for line in code.split('\n') 
            if f"'{docstring_prefix.lower()}" in line.lower()
        ]
        
        if not matching_lines:
            raise ValueError(f"Code must contain '{cls.OUTPUT_DOCSTRING_PREFIX}' docstring. ")
        
        docstring_line = matching_lines[0]
        prefix_idx = docstring_line.index(docstring_prefix) + len(docstring_prefix)
        variables_str = docstring_line[prefix_idx:].strip()
        
        # Return empty list if no variables specified
        if not variables_str:
            return []
        
        # Split by comma and strip whitespace
        return [var.strip("'").strip() for var in variables_str.split(',') if var.strip()]

    @classmethod
    def _validate_return_consistency(cls, tree: ast.AST) -> int:
        """
        Validate that all return statements return the same number of values.
        
        Parameters
        ----------
        tree : ast.AST
            Parsed abstract syntax tree of the code.
            
        Returns
        -------
        int
            The number of return values (0 for empty returns, 1+ for value returns).
            
        Raises
        ------
        ValueError
            If return statements have inconsistent return counts.
        """
        return_values_count = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                current_count = 0
                
                # Returning multiple values (tuple)
                if isinstance(node.value, ast.Tuple):
                    current_count = len(node.value.elts)
                # Returning one value
                elif node.value is not None:
                    current_count = 1
                # Empty return statement
                else:
                    current_count = 0
                
                # Check consistency with previous returns
                if return_values_count is not None and return_values_count != current_count:
                    raise ValueError(
                        "Format Error: all return statements should return the same amount of objects"
                    )
                
                return_values_count = current_count
        
        return return_values_count if return_values_count is not None else 0

    @classmethod
    def _validate_output_docstring(cls, code: str, tree: ast.AST):
        """
        Validate that the Output docstring exists and matches return statements.
        
        Parameters
        ----------
        code : str
            Python code to validate.
        tree : ast.AST
            Parsed abstract syntax tree of the code.
            
        Raises
        ------
        ValueError
            If Output docstring is missing or doesn't match return statements.
        """
        
        # Extract output variables from docstring
        output_variables = cls._extract_docstring_variables(code, cls.OUTPUT_DOCSTRING_PREFIX)
        
        # Get return values count from return statements
        return_values_count = cls._validate_return_consistency(tree)
        
        # Validate that counts match
        if return_values_count != len(output_variables):
            raise ValueError(
                "Format Error: Output docstring does not have the same amount of variables "
                "as the return statements. Ensure the amount of values in output docstring "
                "matches the amount of objects returned."
            )

    @classmethod
    def _validate_dependency_docstring(cls, code: str):
        """
        Validate that the DependentPackages docstring exists and includes all imports.
        
        Handles packages with different import and install names (e.g., sklearn vs 
        scikit-learn). Accepts either the import name or install name in the docstring.
        
        Parameters
        ----------
        code : str
            Python code to validate.
            
        Raises
        ------
        ValueError
            If DependentPackages docstring is missing or incomplete.
        """
        # Check if DependentPackages docstring exists
        if f"'{cls.DEPENDENT_PACKAGES_DOCSTRING_PREFIX}" not in code:
            raise ValueError(
                f"Code must contain '{cls.DEPENDENT_PACKAGES_DOCSTRING_PREFIX}' docstring. "
                f"Use '{cls.DEPENDENT_PACKAGES_DOCSTRING_PREFIX}' for no dependencies or "
                f"'{cls.DEPENDENT_PACKAGES_DOCSTRING_PREFIX} pkg1, pkg2' for dependencies."
            )
        
        # Auto-detect dependencies from imports
        detected_dependencies = cls._auto_detect_dependencies(code)
        
        # Extract dependencies from docstring
        docstring_dependencies = cls._extract_docstring_variables(
            code, cls.DEPENDENT_PACKAGES_DOCSTRING_PREFIX
        )
        
        # Normalize docstring dependencies: map install names back to import names
        # This allows users to specify either import or install names
        reverse_mapping = {v: k for k, v in IMPORT_TO_INSTALL_MAPPING.items()}
        normalized_docstring_deps = set()
        
        for dep in docstring_dependencies:
            # If it's an install name, convert to import name; otherwise keep as-is
            import_name = reverse_mapping.get(dep, dep)
            normalized_docstring_deps.add(import_name)
        
        # Check if all detected dependencies are listed in docstring
        dependency_differences = set(detected_dependencies).difference(normalized_docstring_deps)
        
        if dependency_differences:
            # Provide helpful error message with install names where applicable
            missing_deps_with_install_names = []
            for dep in sorted(dependency_differences):
                install_name = IMPORT_TO_INSTALL_MAPPING.get(dep, dep)
                if install_name != dep:
                    missing_deps_with_install_names.append(f"'{install_name}' (imported as '{dep}')")
                else:
                    missing_deps_with_install_names.append(f"'{dep}'")
            
            raise ValueError(
                f"Format Error: DependentPackages docstring is missing dependencies: "
                f"{', '.join(missing_deps_with_install_names)}. "
                "Ensure all imported packages are listed in the DependentPackages docstring."
            )

    @classmethod
    def validate_id_code(
        cls,
        code: Union[str, Path],
        validate_code: bool = True
    ) -> str:
        """
        Validate and prepare a Python code file for SAS Intelligent Decisioning.
        
        This method validates that the provided code follows the ID format requirements:
        - Must have a function named 'execute'
        - Must include 'Output:' docstring (can be empty: 'Output:')
        - Output docstring must have same amount of variables as the return statements inside of the function.
        - Must include 'DependentPackages:' docstring (can be empty: 'DependentPackages:')
        
        Parameters
        ----------
        code : str or pathlib.Path
            Python code as a string or path to a Python file. The code should
            already be formatted for ID with an execute function and proper docstrings.
        validate_code : bool
            If this boolean is false docstring and syntax validation will be disabled, 
            all that will be done is ensuring the code is imported correctly (reading 
            file/string).
            
        Returns
        -------
        str
            The validated Python code file content.
            
        Raises
        ------
        ValueError
            If code is empty, doesn't contain required docstrings, or is invalid.
        SyntaxError
            If the provided code has syntax errors.
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
        
        if validate_code is False:
            return code
        
        # Validate Python syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Invalid Python syntax in provided code: {e}")
        
        # Validate that it contains an execute function definition
        has_execute_function = any(
            node.name == cls.EXECUTE_FUNCTION_NAME 
            for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )
        if not has_execute_function:
            raise ValueError(f"Code must contain an '{cls.EXECUTE_FUNCTION_NAME}' function")
        
        # Validate Output docstring and return statements
        cls._validate_output_docstring(code, tree)
        
        # Validate DependentPackages docstring
        cls._validate_dependency_docstring(code)
        
        return code

    @classmethod
    def write_id_code_file(
        cls,
        code: Union[str, Path],
        file_name: str,
        folder: Union[str, dict],
        validate_code: bool = True
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
            This boolean flag can be used to disable code validation. The Docstring 
            and syntax of the code will not be checked if false.
            
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
        if not file_name.endswith('.py'):
            raise ValueError("file_name must end with .py extension")
        
        # Validate the code format
        validated_code = cls.validate_id_code(code, validate_code)

        # Upload the file to Viya Files service
        file_obj = file_service.create_file(
            file=validated_code.encode('utf-8'),
            folder=folder,
            filename=file_name,
        )

        data = {
            "name": file_name,
            "fileUri": f"/files/files/{file_obj.id}",
            "type": "decisionPythonFile"
        }
        
        code_file = cls.post("/codeFiles", json=data)
        
        return code_file
