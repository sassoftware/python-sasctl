# Copyright (c) 2024, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List
from pathlib import Path
import re
import importlib.util
import inspect
import importlib
import inspect
import sys
import os
import textwrap
import ast


class ScoreWrapper:
    score_wrapper: str = ""

    @classmethod
    def write_score_wrapper_function_input(cls,
                                           imports: List[str],
                                           function_definition: str,
                                           function_body: str,
                                           model_load: str,
                                           model_name_with_file_extension: str,
                                           output_variables: List[str]):
        """
        Method to generate scoring code from a function and add it to cls.score_wrapper.

        Parameters:
        imports (List[str]): List of modules to import.
        function_definition (str): Function definition.
        function_body (str): Function body.
        model_load (str): Name of the model to load.
        model_name_with_file_extension (str): Name of the model file with extension.
        output_variables (List[str]): List of output variables to define in score function

        Returns:
        cls.score_wrapper (str): The scoring code.
        """
        # Import Modules
        for module in imports:
            cls.score_wrapper += f"import {module}\n"
        # Import pathlib and settings
        cls.score_wrapper += "from pathlib import Path\n"
        cls.score_wrapper += "import settings\n\n\n"

        # Load the model
        cls.score_wrapper += f"model_path = Path(settings.pickle_path) / '{model_name_with_file_extension}'\n\n"
        cls.score_wrapper += f"with open(Path(settings.pickle_path) / '{model_name_with_file_extension}', 'rb') as f:\n\tmodel = {model_load}\n\n"

        # Define the score function and add the function body specified
        cls.score_wrapper += f"{function_definition}:\n"
        cls.score_wrapper += '\t"'
        cls.score_wrapper += "Output Variables: " + ", ".join(output_variables)  # Join output variables with comma
        cls.score_wrapper += '"\n'
        cls.score_wrapper += "\tglobal model\n"
        cls.score_wrapper += "\ttry:\n"
        cls.score_wrapper += f"\t\t{function_body}\n"
        cls.score_wrapper += "\texcept Exception as e:\n"
        cls.score_wrapper += "\t\tprint(f'Error: {e}')\n"
        cls.score_wrapper += "\t\treturn None\n"

        # Validate syntax before returning
        if not cls.validate_score_wrapper_syntax(cls.score_wrapper):
            raise SyntaxError("Syntax error in generated code.")

        return cls.score_wrapper

    @classmethod
    def write_score_wrapper_file_input(cls,
                                       imports: List[str],
                                       file_path: str,
                                       model_load: str,
                                       model_name_with_file_extension: str,
                                       score_function_body: str,
                                       output_variables: List[str],
                                       ):
        """
        Method to generate scoring code from a file and add it to cls.score_wrapper.

        Parameters:
        imports (List[str]): List of modules to import.
        file_path (str): Path to the file containing the additional classes and functions necessary for scoring.
        model_load (str): Name of the model to load.
        model_name_with_file_extension (str): Name of the model file with extension.
        score_function_body (str): The code needed to evaluate the model.
        output_variables (List[str]): List of output variables to define in score function

        Returns:
        cls.score_wrapper (str): The scoring code.
        """
        # write imports to file
        for module in imports:
            cls.score_wrapper += f"import {module}\n"
        cls.score_wrapper += "from pathlib import Path\n"
        cls.score_wrapper += "import settings\n\n\n"

        # open a file and read all of the class definitions and functions, and write them to cls.score_wrapper maintain spacing of the code being written
        with open(file_path, "r") as file:
            code = file.read()
            cls.score_wrapper += code

        # load model
        cls.score_wrapper += f"\n\nmodel_path = Path(settings.pickle_path) / '{model_name_with_file_extension}'\n"
        cls.score_wrapper += f"{model_load}\n\n"

        # define the generic score function, and append the score_function_body to evaluate the model.
        cls.score_wrapper += f"def score(input_data):\n"
        cls.score_wrapper += '\t"'
        cls.score_wrapper += "Output Variables: " + ", ".join(output_variables)  # Join output variables with comma
        cls.score_wrapper += '"\n'

        cls.score_wrapper += "\tglobal model\n"
        cls.score_wrapper += "\ttry:"
        cls.score_wrapper += f"\n{score_function_body}\n"
        cls.score_wrapper += "\n\texcept Exception as e:\n"
        cls.score_wrapper += "\t\tprint(f'Error: {e}')\n"
        cls.score_wrapper += "\t\treturn None\n"
        # Need some kind of return value here
        return cls.score_wrapper

        # Validate Syntax before returning
        if not cls.validate_score_wrapper_syntax(cls.score_wrapper):
            raise SyntaxError("Syntax error in generated code.")

        return cls.score_wrapper

    @classmethod
    def write_wrapper_to_file(cls,
                              path: str,
                              model_prefix: str,
                              file_name: str) -> None:
        """
        Method to write the score wrapper to a file.

        Parameters:
        file_name (str): Name of the file to write the score wrapper to.

        Returns:
        None
        """
        score_wrapper_path = Path(path) / f"score_{model_prefix}.py"
        with open(score_wrapper_path, "w") as score_wrapper_file:
            score_wrapper_file.write(cls.score_wrapper)

    @classmethod
    def validate_score_wrapper_syntax(cls, code: str) -> bool:
        """
        Method to perform a syntax check on the provided code.

        Parameters:
        code (str): Code to be checked for syntax.

        Returns:
        bool: True if syntax is correct, False otherwise.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False



