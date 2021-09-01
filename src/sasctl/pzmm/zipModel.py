# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
from pathlib import Path
import zipfile
import io

# %%

class ZipModel():
    
    @staticmethod
    def zipFiles(fileDir, modelPrefix):
        '''
        Combines all JSON files with the model pickle file and associated score code file
        into a single archive ZIP file.
        
        Parameters
        ---------------
        fileDir : string
            Location of *.json, *.pickle, and *Score.py files.
        modelPrefix : string
            Variable name for the model to be displayed in SAS Open Model Manager 
            (i.e. hmeqClassTree + [Score.py || .pickle]).
            
        Yields
        ---------------
        '*.zip'
            Archived ZIP file for importing into SAS Open Model Manager. In this form,
            the ZIP file can be imported into SAS Open Model Manager.
        '''
        
        fileNames = []
        fileNames.extend(sorted(Path(fileDir).glob('*.json')))
        fileNames.extend(sorted(Path(fileDir).glob('*Score.py')))
        fileNames.extend(sorted(Path(fileDir).glob('*.pickle')))
        # Include H2O.ai MOJO files 
        fileNames.extend(sorted(Path(fileDir).glob('*.mojo')))
        
        with zipfile.ZipFile(Path(fileDir) / (modelPrefix + '.zip'), mode='w') as zFile:
            for file in fileNames:
                zFile.write(file, arcname=file.name)
                
        with open(Path(fileDir) / (modelPrefix + '.zip'), 'rb') as zipFile:
            return io.BytesIO(zipFile.read())
