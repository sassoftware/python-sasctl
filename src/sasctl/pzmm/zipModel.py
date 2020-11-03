# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
from pathlib import Path
import zipfile
import io

# %%

class ZipModel():
    
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
        
        with zipfile.ZipFile(Path(fileDir) / (modelPrefix + '.zip'), mode='w') as zFile:
            for name in fileNames:
                zFile.write(name, arcname=name)
                
        with open(Path(fileDir) / (modelPrefix + '.zip'), 'rb') as zipFile:
<<<<<<< HEAD
            return io.BytesIO(zipFile.read())
=======
            return io.BytesIO(zipFile.read())
>>>>>>> ca5a7c6364d6e16a6dac27176dc0b975965ea57a
