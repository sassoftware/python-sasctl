# Copyright (c) 2024, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List
from pathlib import Path
import re

class ScoreWrapper:
    score_wrapper: str = ""

    @classmethod
    def write_score_wrapper_function_input(cls, 
                                           imports: List[str], 
                                           function_definition: str, 
                                           function_body: str, 
                                           model_load: str, 
                                           model_name_with_file_extension: str):
        """
        Method to generate scoring code from a function and add it to cls.score_wrapper.

        Parameters:
        imports (List[str]): List of modules to import.
        function_definition (str): Function definition code.
        function_body (str): Function body code.
        model_load (str): Name of the model to load.
        model_name_with_file_extension (str): Name of the model file with extension.

        Returns:
        cls.score_wrapper (str): The scoring code.
        """
        for module in imports:
            cls.score_wrapper += f"import {module}\n"
        
        cls.score_wrapper += "from pathlib import Path\n"
        cls.score_wrapper += "import settings\n\n\n"

        cls.score_wrapper += f"model_path = Path(settings.pickle_path) / '{model_name_with_file_extension}'\n\n"
        cls.score_wrapper += f"with open(Path(settings.pickle_path) / '{model_name_with_file_extension}', 'rb') as f:\n\tmodel = {model_load}\n\n"

        cls.score_wrapper += f"{function_definition}:\n"
        cls.score_wrapper += "\tglobal model\n"
        cls.score_wrapper += "\ttry:\n"
        cls.score_wrapper += f"\t\t{function_body}\n"
        cls.score_wrapper += "\texcept Exception as e:\n"
        cls.score_wrapper += "\t\tprint(f'Error: {e}')\n"
        return cls.score_wrapper

    @classmethod
    def write_score_wrapper_file_input(cls, 
                                       imports: List[str], 
                                       file_path: str, 
                                       model_load: str, 
                                       model_scoring_code: str, 
                                       model_name_with_file_extension: str):
        """
        Method to generate scoring code from a file and add it to cls.score_wrapper.

        Parameters:
        imports (List[str]): List of modules to import.
        file_path (str): Path to the file containing the scoring code.
        model_load (str): Name of the model to load.
        model_scoring_code (str): Model scoring code. How the function is called.
        model_name_with_file_extension (str): Name of the model file with extension.

        Returns:
        cls.score_wrapper (str): The scoring code.
        """
        for module in imports:
            cls.score_wrapper += f"import {module}\n"
        
        cls.score_wrapper += "from pathlib import Path\n"
        cls.score_wrapper += "import settings\n\n\n"

        cls.score_wrapper += f"model_path = Path(settings.pickle_path) / '{model_name_with_file_extension}'\n\n"
        cls.score_wrapper += f"with open(Path(settings.pickle_path) / '{model_name_with_file_extension}', 'rb') as f:\n\tmodel = {model_load}\n\n"

        with open(file_path, 'r') as f:
            file_content = f.read()
        function_defs = re.findall(r"(def .+?:.*?)(?=def |\Z)", file_content, re.DOTALL)
        for function in function_defs:
            cls.score_wrapper += f"{function}\n"
        
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


