#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import json
import pandas as pd

from pathlib import Path


def convertMetadata(zPath, pythonScoreCode=None):
    """Modify the model metadata to match new requirements in SAS Viya 4. The
    scoreCodeType in the ModelProperties.json file should be labelled as 'Python'.
    The file associated with the 'score' role should be a Python file.

    Parameters
    ----------
    zPath : string or Path object
        Location of files in the SAS Viya 3.5 model zip.
    pythonScoreCode : string, optional
        File name of the Python score code. If None, then the name is
        determined by the files in the model zip. Default value is None.

    Returns
    -------
    scoreResource : list
        File name(s) of the score resource file.
    pythonScoreCode : string
        File name of the Python score code file.

    Raises
    ------
    SyntaxError :
        If no pythonScoreCode name is provided, but there are multiple Python
        files in the zPath, then a SyntaxError is raised asking for further
        clarification as to which file is the Python score code.
    """
    # Replace the value of scoreCodeType to 'Python' in ModelProperties.json
    with open(Path(zPath) / "ModelProperties.json", "r") as jFile:
        modelProperties = json.loads(jFile.read())

    modelProperties.update({"scoreCodeType": "Python"})
    propIndex = list(modelProperties.keys())
    propValues = list(modelProperties.values())
    outputJSON = pd.Series(propValues, index=propIndex)

    with open(Path(zPath) / "ModelProperties.json", "w") as jFile:
        dfDump = pd.Series.to_dict(outputJSON.transpose())
        json.dump(dfDump, jFile, indent=4, skipkeys=True)
        print("ModelProperties.json has been modified and rewritten for SAS Viya 4")

    # Replace the 'score' role file with Python score code file
    with open(Path(zPath) / "fileMetaData.json", "r") as jFile:
        metaData = json.loads(jFile.read())
    if pythonScoreCode is None:
        numPyFiles = 0
        for file in zPath.glob("*.py"):
            pythonScoreCode = file.name
            numPyFiles = numPyFiles + 1
        if numPyFiles > 1:
            message = (
                "More than one Python file was found, therefore the score code"
                + " the score code could not be determined. Please provide the"
                + " name of the Python score code file as an argument."
            )
            raise SyntaxError(message)
    scoreResource = []
    for i in range(len(metaData)):
        if metaData[i]["role"] == "score":
            metaData[i].update({"name": pythonScoreCode})
        if metaData[i]["role"] == "scoreResource":
            scoreResource.append(metaData[i]["name"])
        if metaData[i]["role"] == "python pickle":
            scoreResource.append(metaData[i]["name"])
    with open(Path(zPath) / "fileMetaData.json", "w") as jFile:
        json.dump(metaData, jFile, indent=4, skipkeys=True)
        print("fileMetaData.json has been modified and rewritten for SAS Viya 4")

    return scoreResource, pythonScoreCode


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
    scoreResource, pythonScoreCode = convertMetadata(zPath, pythonScoreCode=None)
    convertScoreCode(zPath, scoreResource, pythonScoreCode)
