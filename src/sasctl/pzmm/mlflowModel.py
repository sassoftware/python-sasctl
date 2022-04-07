# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import json


class MLFlowModel:
    def readMLmodelFile(self, mPath=Path.cwd()):
        with open(Path(mPath) / "MLmodel", "r") as mFile:
            mLines = mFile.readlines()

        # More verbose substring acceptance is needed for each possible model type
        # For now, stick with those models which are based on pickle files and don't report model type or version
        # ind = self.findSubstringIndex(mLines, 'loader_module')
        # package = mLines[ind[0]].strip().split(' ')[1].split('.')[1]

        # ind = self.findSubstringIndex(mLines, package + '_version')
        # packageVersion = mLines[ind[0]].strip().split(' ')[1]

        varList = ["python_version", "serialization_format", "run_id", "model_path"]
        for i, varString in enumerate(varList):
            index = [i for i, s in enumerate(mLines) if varString in s]
            if not index:
                raise ValueError("This MLFlow model type is not currently supported.")
            varList[i] = {varList[i]: mLines[index[0]].strip().split(" ")[1]}

        varDict = {k: v for d in varList for k, v in d.items()}
        varDict["mlflowPath"] = mPath

        indIn = [i for i, s in enumerate(mLines) if "inputs:" in s]
        indOut = [i for i, s in enumerate(mLines) if "outputs:" in s]

        inputs = mLines[indIn[0] : indOut[0]]
        outputs = mLines[indOut[0] : -1]

        inputsDict = json.loads("".join([s.strip() for s in inputs])[9:-1])
        outputsDict = json.loads("".join([s.strip() for s in outputs])[10:-1])

        return varDict, inputsDict, outputsDict  # ,package, packageVersion
