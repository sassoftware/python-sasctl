# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from ..core import platform_version
from .._services.model_repository import ModelRepository as mr
from .writeScoreCode import ScoreCode as sc
from .zipModel import ZipModel as zm

class ImportModel():
    
    @classmethod
    def pzmmImportModel(cls, zPath, modelPrefix, project, inputDF, targetDF, predictmethod,
                          metrics=['EM_EVENTPROBABILITY', 'EM_CLASSIFICATION'], 
                          modelFileName=None, pyPath=None, threshPrediction=None,
                          otherVariable=False, isH2OModel=False, force=False):
        '''Import model to SAS Model Manager using pzmm submodule.
        
        Using pzmm, generate Python score code and import the model files into 
        SAS Model Manager. This function automatically checks the version of SAS
        Viya being used through the sasctl Session object and creates the appropriate
        score code and API calls required for the model and its associated content to 
        be registered in SAS Model Manager.

        Parameters
        ----------
        zPath : string or Path
            Directory location of the files to be zipped and imported as a model.
        modelPrefix : string
            The variable for the model name that is used when naming model files.
            (For example: hmeqClassTree + [Score.py || .pickle]).
        project : str or dict
            The name or id of the model project, or a dictionary
            representation of the project.
        inputDF : DataFrame
            The `DataFrame` object contains the training data, and includes only the predictor
            columns. The writeScoreCode function currently supports int(64), float(64),
            and string data types for scoring.
        targetDF : DataFrame
            The `DataFrame` object contains the training data for the target variable.
        predictMethod : string
            User-defined prediction method for score testing. This should be
            in a form such that the model and data input can be added using 
            the format() command. 
            For example: '{}.predict_proba({})'.
        metrics : string list, optional
            The scoring metrics for the model. The default is a set of two
            metrics: EM_EVENTPROBABILITY and EM_CLASSIFICATION.        
        modelFileName : string, optional
            Name of the model file that contains the model. By default None and assigned as
            modelPrefix + '.pickle'.
        pyPath : string, optional
            The local path of the score code file. By default None and assigned as the zPath.
        threshPrediction : float, optional
            The prediction threshold for probability metrics. For classification,
            below this threshold is a 0 and above is a 1.
        otherVariable : boolean, optional
            The option for having a categorical other value for catching missing
            values or values not found in the training data set. The default setting
            is False.
        isH2OModel : boolean, optional
            Sets whether the model is an H2O.ai Python model. By default False.
        force : boolean, optional
            Sets whether to overwrite models with the same name upon upload. By default False.
            
        Yields
        ------
        '*Score.py'
            The Python score code file for the model.
        '*.zip'
            The zip archive of the relevant model files. In Viya 3.5 the Python score
            code is not present in this initial zip file.
        '''
        noScoreCode = False
        binaryModel = False
        if pyPath is None:
            pyPath = Path(zPath)
        else:
            pyPath = Path(pyPath)
            
        def getFiles(extensions):
            allFiles = []
            for ext in extensions:
                allFiles.extend(pyPath.glob(ext))
            return allFiles
        import pdb; pdb.set_trace()
        if modelFileName is None:
            if isH2OModel:
                binaryOrMOJO = getFiles(['*.mojo', '*.pickle'])
                if len(binaryOrMOJO) == 0:
                    print('WARNING: An H2O model file was not found at {}. Score code will not be automatically generated.'.format(str(pyPath)))
                    noScoreCode = True
                elif len(binaryOrMOJO) == 1:
                    if str(binaryOrMOJO[0]).endswith('.pickle'):
                        binaryModel = True
                        modelFileName = modelPrefix + '.pickle'
                    else:
                        modelFileName = modelPrefix + '.mojo'
                else:
                    print('WARNING: Both a MOJO and binary model file are present at {}. Score code will not be automatically generated.'.format(str(pyPath)))
                    noScoreCode = True
            else:
                modelFileName = modelPrefix + '.pickle'
                
        isViya35 = (platform_version() == '3.5')
        if not isViya35:
            if noScoreCode:
                print('No score code was generated.')
            else:
                sc.writeScoreCode(inputDF, targetDF, modelPrefix, predictmethod, modelFileName,
                                  metrics=metrics, pyPath=pyPath, threshPrediction=threshPrediction, 
                                  otherVariable=otherVariable, isH2OModel=isH2OModel, isBinaryModel=binaryModel)
                print('Model score code was written successfully to {}.'.format(Path(pyPath) / (modelPrefix + 'Score.py')))
            zipIOFile = zm.zipFiles(Path(zPath), modelPrefix)
            print('All model files were zipped to {}.'.format(Path(zPath)))
            response = mr.import_model_from_zip(modelPrefix, project, zipIOFile, force)
            try:
                print('Model was successfully imported into SAS Model Manager as {} with UUID: {}.'.format(response.name, response.id))
            except AttributeError:
                print('Model failed to import to SAS Model Manager.')
        else:
            zipIOFile = zm.zipFiles(Path(zPath), modelPrefix)
            print('All model files were zipped to {}.'.format(Path(zPath)))
            response = mr.import_model_from_zip(modelPrefix, project, zipIOFile, force)
            try:
                print('Model was successfully imported into SAS Model Manager as {} with UUID: {}.'.format(response.name, response.id))
            except AttributeError:
                print('Model failed to import to SAS Model Manager.')
            if noScoreCode:
                print('No score code was generated.')
            else:
                sc.writeScoreCode(inputDF, targetDF, modelPrefix, predictmethod, modelFileName,
                                  metrics=metrics, pyPath=pyPath, threshPrediction=threshPrediction,
                                  otherVariable=otherVariable, model=response.id,
                                  isH2OModel=isH2OModel, isBinaryModel=binaryModel)
                print('Model score code was written successfully to {} and uploaded to SAS Model Manager'.format(Path(pyPath) / (modelPrefix + 'Score.py')))