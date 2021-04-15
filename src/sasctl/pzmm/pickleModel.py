# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# %%
from pathlib import Path

import pickle
import gzip

# %%
class PickleModel():
    
    def pickleTrainedModel(self, trainedModel, modelPrefix, pPath=Path.cwd(), isH2OModel=False):
        '''
        Write trained model to a binary pickle file. 
        
        Parameters
        ---------------
        trainedModel : model or string or Path
            For non-H2O models, this argument contains the model variable. Otherwise,
            this should be the file path of the MOJO file.
        modelPrefix : string
            Variable name for the model to be displayed in SAS Open Model Manager 
            (i.e. hmeqClassTree + [Score.py || .pickle]).
        pPath : string, optional
            File location for the output pickle file. Default is the current
            working directory.
        isH2OModel : boolean, optional
            Sets whether the model file is an H2O.ai MOJO file. If set as True, 
            the MOJO file will be gzipped before uploading to SAS Model Manager.
            The default value is False.
			
		Yields
		---------------
		'*.pickle'
			Binary pickle file containing a trained model.
        '*.mojo'
            Archived H2O.ai MOJO file containing a trained model.
        '''
        
        if not isH2OModel:
            with open(Path(pPath) / (modelPrefix + '.pickle'), 'wb') as pFile:
                pickle.dump(trainedModel, pFile)
        else:
            with open(Path(trainedModel), 'rb') as fileIn, gzip.open(Path(pPath) / (modelPrefix + '.mojo'), 'wb') as fileOut:
                fileOut.writelines(fileIn)