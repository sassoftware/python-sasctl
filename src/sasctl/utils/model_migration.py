#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path


def convert_metadata(zip_path, python_score_code=None):
    """Modify SAS Viya 3.5 model metadata to match new requirements in SAS Viya 4.

    The scoreCodeType in the ModelProperties.json file should be labelled as 'Python'.
    The file associated with the 'score' role should be a Python file.

    Parameters
    ----------
    zip_path : string or Path object
        Location of files in the SAS Viya 3.5 model zip directory.
    python_score_code : string, optional
        File name of the Python score code. If None, then the file name is assumed to be
        the only Python file in the zip_path. An error is raised if this is not the
        case. Default value is None.

    Returns
    -------
    score_resource : list
        File name(s) of the score resource file.
    python_score_code : string
        File name of the Python score code file.

    Raises
    ------
    ValueError
        If no python_score_code name is provided, but there are multiple Python
        files in the zip_path, then a SyntaxError is raised asking for further
        clarification as to which file is the Python score code.
    """
    # Replace the value of scoreCodeType to 'Python' in ModelProperties.json
    with open(Path(zip_path) / "ModelProperties.json", "r") as file:
        model_properties = json.loads(file.read())

    model_properties.update({"scoreCodeType": "Python"})

    with open(Path(zip_path) / "ModelProperties.json", "w") as file:
        json.dump(model_properties, file, indent=4)
        print("ModelProperties.json has been modified and rewritten for SAS Viya 4")

    # Replace the 'score' role file with Python score code file
    with open(Path(zip_path) / "fileMetaData.json", "r") as file:
        meta_data = json.loads(file.read())
    if python_score_code is None:
        if len(list(zip_path.glob("*.py"))) > 1:
            raise ValueError(
                f"More than one Python file was found at {zip_path}, "
                f"therefore the score code file could not be determined. "
                f"Please provide the name of the Python score code file "
                f"as an argument."
            )
        else:
            python_score_code = list(zip_path.glob("*.py"))[0].name

    score_resource = []
    for i in range(len(meta_data)):
        if meta_data[i]["role"] == "score":
            meta_data[i].update({"name": python_score_code})
        if meta_data[i]["role"] == "scoreResource":
            score_resource.append(meta_data[i]["name"])
        if meta_data[i]["role"] == "python pickle":
            score_resource.append(meta_data[i]["name"])
    with open(Path(zip_path) / "fileMetaData.json", "w") as file:
        json.dump(meta_data, file, indent=4, skipkeys=True)
        print("fileMetaData.json has been modified and rewritten for SAS Viya 4")

    return score_resource, python_score_code


def convert_score_code(zip_path, score_resource, python_score_code):
    """Convert the Python score code used in SAS Viya 3.5 into score code usable in
    SAS Viya 4.

    The two main adjustments are including an 'import settings' statement and
    replacing score resource access calls that reference an explicit path with
    'settings.pickle_path' + <score_resource>.

    Parameters
    ----------
    zip_path : string or Path object
        Location of files in the SAS Viya 3.5 model zip.
    score_resource : list
        File name(s) of the score resource file.
    python_score_code : string
        File name of the Python score code file.
    """
    # Read entire text of score code
    with open(Path(zip_path) / python_score_code, "r") as pyFile:
        score_code = pyFile.read()

    # Add the import settings line to the score code
    score_code = "import settings\n" + score_code

    # Search for all directory paths in score code that contain the score_resource
    old_string = re.findall(r"['\"]\/.*?\.[\w:]+['\"]", score_code)
    parsed_old_string = []
    new_string = []
    for resource in score_resource:
        string_found = [s for s in old_string if resource in s]
        parsed_old_string = parsed_old_string + string_found
        if string_found:
            new_string.append(f"settings.pickle_path + '{resource}'")
    for old, new in zip(parsed_old_string, new_string):
        score_code = score_code.replace(str(old), str(new))

    # Write new text of score code to file
    with open(Path(zip_path) / python_score_code, "w") as pyFile:
        pyFile.write(score_code)
        print(f"{python_score_code} has been modified and rewritten for SAS Viya 4")


def delete_sas_files(zip_path):
    """Remove .sas score files created for SAS Viya 3.5, which are no longer
    needed in SAS Viya 4.

    These files are typically named score.sas, dmcas_packagescorecode.sas, or
    dmcas_epscorescode.sas.

    Parameters
    ----------
    zip_path : string or Path object
        Location of files in the SAS Viya 3.5 model zip.
    """
    zip_path = Path(zip_path)
    for file in zip_path.glob("*.sas"):
        file.unlink()


def convert_model_zip(zip_path, python_score_code=None):
    """Pass the directory path of the model to be converted from SAS Viya 3.5 to
    SAS Viya 4.
    This will remove any .sas files, adjust the ModelProperties.json and
    fileMetaData.json files, and modify the score code.

    Parameters
    ----------
    zip_path : string or Path object
        Location of files in the SAS Viya 3.5 model zip.
    python_score_code : string, optional
        File name of the Python score code. If None, then the name is determined by the
        files in the model zip. Default value is None.
    """
    delete_sas_files(zip_path)
    score_resource, python_score_code = convert_metadata(zip_path, python_score_code)
    convert_score_code(zip_path, score_resource, python_score_code)
