# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from pathlib import Path


class PickleModel:
    @staticmethod
    def pickleTrainedModel(trainedModel, modelPrefix, pPath=Path.cwd()):
        """Write trained model to a binary pickle file.
        
        Parameters
        ----------
        trainedModel
            User-defined trained model.
        modelPrefix : string
            Variable name for the model to be displayed in SAS Open Model Manager 
            (i.e. hmeqClassTree + [Score.py || .pickle]).
        pPath : pathlib.Path, optional
            File location for the output pickle file. Default is the current
            working directory.

        Returns
        -------
        None

        """
        
        with open(pPath / (modelPrefix + '.pickle'), 'wb') as pFile:
            pickle.dump(trainedModel, pFile)
