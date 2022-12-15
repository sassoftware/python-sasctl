#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import json
import pandas as pd

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
    SyntaxError
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
            raise ValueError(f"More than one Python file was found at {zip_path}, "
                             f"therefore the score code file could not be determined. "
                             f"Please provide the name of the Python score code file "
                             f"as an argument.")
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


def convertScoreCode(zPath, scoreResource, pythonScoreCode):
    """Convert the Python score code used in SAS Viya 3.5 into score code usable in
    SAS Viya 4. The two adjustments are including an 'import settings' statement and
    replacing score resource access calls that reference an explicit path with
    'settings.pickle_path' + <scoreResource>.

    Parameters
    ----------
    zPath : string or Path object
        Location of files in the SAS Viya 3.5 model zip.
    scoreResource : list
        File name(s) of the score resource file.
    pythonScoreCode : string
        File name of the Python score code file.
    """
    # Read entire text of score code
    with open(Path(zPath) / pythonScoreCode, "r") as pyFile:
        scoreCode = pyFile.read()

    # Add the import settings line to the score code
    scoreCode = "import settings\n" + scoreCode

    # Search for all directory paths in score code that contain the scoreResource
    oldString = re.findall(r"['\"]\/.*?\.[\w:]+['\"]", scoreCode)
    parsedOldString = []
    newString = []
    for resource in scoreResource:
        stringFound = []
        stringFound = [s for s in oldString if resource in s]
        parsedOldString = parsedOldString + stringFound
        if stringFound:
            newString.append("settings.pickle_path + '{}'".format(resource))
    for old, new in zip(parsedOldString, newString):
        scoreCode = scoreCode.replace(str(old), str(new))

    # Write new text of score code to file
    with open(Path(zPath) / pythonScoreCode, "w") as pyFile:
        pyFile.write(scoreCode)
        print(
            "{} has been modified and rewritten for SAS Viya 4".format(pythonScoreCode)
        )


def deleteSASFiles(zPath):
    """Remove .sas score files created for SAS Viya 3.5, which are no longer
    needed in SAS Viya 4. These files are typically named score.sas,
    dmcas_packagescorecode.sas, or dmcas_epscorescode.sas.

    Parameters
    ----------
    zPath : string or Path object
        Location of files in the SAS Viya 3.5 model zip.
    """
    zPath = Path(zPath)
    for file in zPath.glob("*.sas"):
        file.unlink()


def convertModelZip(zPath, pythonScoreCode=None):
    """Pass the directory path of the model to be converted from SAS Viya 3.5 to
    SAS Viya 4. Then the function removes any .sas files, adjusts the ModelProperties.json
    and fileMetaData.json files, and modifies the score code.

    Parameters
    ----------
    zPath : string or Path object
        Location of files in the SAS Viya 3.5 model zip.
    pythonScoreCode : string, optional
        File name of the Python score code. If None, then the name is
        determined by the files in the model zip. Default value is None.
    """
    deleteSASFiles(zPath)
    scoreResource, pythonScoreCode = convert_metadata(zPath, python_score_code=None)
    convertScoreCode(zPath, scoreResource, pythonScoreCode)
