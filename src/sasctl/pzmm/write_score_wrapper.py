# Copyright (c) 2024, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List
from pathlib import Path
import re

"""
- Collect inputs and outputs from inputVar.json and outputVar.json
   - Throw error or include optional arguments
- How to handle imports?
   - pass a list of imports, etc
- How do we handle model loading?
   - re.sub() with "Path(settings.pickle_path) / " and file name
   - Document replacement of path with inclusion of settings.pickle_path
- What form do we want to take the score snippet in? add boolean to check whether user is passing in a function or passing in the score snipped itslef if so adjust the code accordingly
   - function...we'll parse and drop in the contents
   - file, but tell us the score function...include other functions and pull out score function info
"""


class ScoreWrapper:
    score_wrapper: str = ""

    @classmethod
    def write_score_wrapper(cls,
                            imports: List[str],
                            model_load: str,
                            function_definition: str,
                            model_scoring_code: str,
                            is_function: bool,
                            is_file: bool) -> str:
        """
        Method to generate scoring code and add it to cls.score_wrapper.


        Parameters:
        imports (List[str]): List of modules to import.
        model_load (str): Name of the model to load.
        function_definition (str): Function definition code.
        model_scoring_code (str): Model scoring code. how the function is called.

        Returns: cls.score_wrapper (str): Score wrapper code.
        """
        # Adding imports to score_wrapper
        for module in imports:
            cls.score_wrapper += f"import {module}\n"
            cls.score_wrapper += "from pathlib import Path\n"
            cls.score_wrapper += "import settings\n"

        # Adding model load to score_wrapper
        cls.score_wrapper += f"{model_load} = Path(settings.pickle_path) / '{model_load}'\n"

        # Adding function definition to score_wrapper
        if is_function:
            # if the function definition is given.
            cls.score_wrapper += f"{function_definition}\n"
        elif is_file:
            # file, but tell us the score function...include other functions and pull out score function info use re to find all the functions specified maybe add parameters for if is_file is true to know how many functions user wants to include
            # Extract all function definitions from the file
            with open(function_definition, 'r') as f:
                file_content = f.read()
            # This only works for function definitions on one line like this def score(input_data):
            function_defs = re.findall(r"(def .+?:)", file_content)
            for function in function_defs:
                cls.score_wrapper += f"{function}\n"

        # Adding model scoring code inside a try/except block to score_wrapper
        cls.score_wrapper += "try:\n"
        cls.score_wrapper += f"    global {model_load}\n"
        cls.score_wrapper += f"    {model_scoring_code}\n"
        cls.score_wrapper += "except Exception as e:\n"
        cls.score_wrapper += "    print(f'Error: {e}')\n"

        return cls.score_wrapper

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


"""
<imports provided by user>
import settings
<model load>
   add Path(settings.pickle_path
function definition
docstring: output string
globalize model and try/except block
<model scoring code>
clean up output
"""

# example of definition and model scoring code
function_definition = """
def score(input_data):
   # Preprocess data
   processed_data = preprocess(input_data)
   # Score the processed data
   return my_model.predict(processed_data)
"""

model_scoring_code = """
input_data = get_input_data()
score(input_data)
"""


