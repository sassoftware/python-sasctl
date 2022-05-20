# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
from pathlib import Path
import zipfile
import io

# %%


class ZipModel:
    @staticmethod
    def zipFiles(fileDir, modelPrefix, isViya4=False):
        """
        Combines all JSON files with the model pickle file and associated score code file
        into a single archive ZIP file.

        Parameters
        ---------------
        fileDir : string
            Location of *.json, *.pickle, and *Score.py files.
        modelPrefix : string
            Variable name for the model to be displayed in SAS Open Model Manager
            (i.e. hmeqClassTree + [Score.py || .pickle]).
        isViya4 : boolean, optional
            Boolean to indicate difference in logic between SAS Viya 3.5 and SAS Viya 4.
            For Viya 3.5 models, ignore score code that is already in place in the file
            directory provided. Default value is False.

        Yields
        ---------------
        '*.zip'
            Archived ZIP file for importing into SAS Open Model Manager. In this form,
            the ZIP file can be imported into SAS Open Model Manager.
        """

        fileNames = []
        fileNames.extend(sorted(Path(fileDir).glob("*.json")))
        if isViya4:
            fileNames.extend(sorted(Path(fileDir).glob("*Score.py")))
        fileNames.extend(sorted(Path(fileDir).glob("*.pickle")))
        # Include H2O.ai MOJO files
        fileNames.extend(sorted(Path(fileDir).glob("*.mojo")))

        with zipfile.ZipFile(
            str(Path(fileDir) / (modelPrefix + ".zip")), mode="w"
        ) as zFile:
            for file in fileNames:
                zFile.write(str(file), arcname=file.name)

        with open(str(Path(fileDir) / (modelPrefix + ".zip")), "rb") as zipFile:
            return io.BytesIO(zipFile.read())
