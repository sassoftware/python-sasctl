# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import numpy as np
from ..tasks import get_software_version, upload_and_copy_score_resources
from .._services.model_repository import ModelRepository as modelRepo

# %%
class ScoreCode():
    
    @classmethod
    def writeScoreCode(cls, inputDF, targetDF, modelPrefix,
                       predictMethod, modelFileName,
                       metrics=['EM_EVENTPROBABILITY', 'EM_CLASSIFICATION'],
                       pyPath=Path.cwd(), threshPrediction=None,
                       otherVariable=False, model=None, isH2OModel=False, missingValues=False):
        '''
        Writes a Python score code file based on training data used to generate the model 
        pickle file. The Python file is included in the ZIP file that is imported or registered 
        into the common model repository. The model can then be used by SAS applications, 
        such as SAS Open Model Manager.
        
        The score code that is generated is designed to be a working template for any
        Python model, but is not guaranteed to work out of the box for scoring, publishing, 
        or validating the model.
        
        Note that for categorical variables, the variable is split into the possible 
        categorical values of the variable. Also, by default it does NOT include a catch-all 
        [catVar]_Other variable to store any missing values or any values not found in the 
        training data set. If you have missing values or values not included in your training 
        data set, you must set the OtherVariable option to True.
        
        Both the inputDF and targetDF dataframes have the following stipulations:
        * Column names must be a valid Python variable name.
        * For categorical columns, the values must be a valid Python variable name.
        If either of these conditions is broken, an exception is raised.
        
        Parameters
        ----------
        inputDF : DataFrame
            The `DataFrame` object contains the training data, and includes only the predictor
            columns. The writeScoreCode function currently supports int(64), float(64),
            and string data types for scoring.
        targetDF : DataFrame
            The `DataFrame` object contains the training data for the target variable.
        modelPrefix : string
            The variable for the model name that is used when naming model files.
            (For example: hmeqClassTree + [Score.py || .pickle]).      
        predictMethod : string
            User-defined prediction method for score testing. This should be
            in a form such that the model and data input can be added using 
            the format() command. 
            For example: '{}.predict_proba({})'.
        modelFileName : string
            Name of the model file that contains the model.
        metrics : string list, optional
            The scoring metrics for the model. The default is a set of two
            metrics: EM_EVENTPROBABILITY and EM_CLASSIFICATION.
        pyPath : string, optional
            The local path of the score code file. The default is the current
            working directory.
        threshPrediction : float, optional
            The prediction threshold for probability metrics. For classification,
            below this threshold is a 0 and above is a 1.
        otherVariable : boolean, optional
            The option for having a categorical other value for catching missing
            values or values not found in the training data set. The default setting
            is False.
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model. The default value is None and is only necessary for models that
            will be hosted on SAS Viya 3.5.
        isH2OModel : boolean, optional
            Sets whether the model is an H2O.ai Python model. By default False.
        missingValues : boolean, optional
            Sets whether data used for scoring needs to go through imputation for
            missing values before passed to the model. By default false.
    			
    	Yields
    	------
        '*Score.py'
            The Python score code file for the model.
        '''       
        # Call REST API to check SAS Viya version
        isViya35 = (get_software_version() == '3.5')
        
        # Initialize modelID to remove unbound variable warnings
        modelID = None
        
        # For SAS Viya 3.5, either return an error or return the model UUID as a string
        if isViya35:
            if model == None:
                raise ValueError('The model UUID is required for score code written for' +
                                 ' SAS Model Manager on SAS Viya 3.5.')
            elif modelRepo.is_uuid(model):
                modelID = model
            elif isinstance(model, dict) and 'id' in model:
                modelID = model['id']
            else:
                model = modelRepo.get_model(model)
                modelID = model['id']
        
        # From the input dataframe columns, create a list of input variables, then check for viability
        inputVarList = list(inputDF.columns)
        for name in inputVarList:
            if not str(name).isidentifier():
                raise SyntaxError('Invalid column name in inputDF. Columns must be ' +
                                  'valid as Python variables.')
        newVarList = list(inputVarList)
        inputDtypesList = list(inputDF.dtypes)        
    
        # Set the location for the Python score file to be written, then open the file
        pyPath = Path(pyPath) / (modelPrefix + 'Score.py')
        with open(pyPath, 'w') as cls.pyFile:
        
            # For H2O models, include the necessary packages
            if isH2OModel:
                cls.pyFile.write('''\
import h2o
import gzip, shutil, os''')
            # Import math for imputation; pickle for serialized models; pandas for data management; numpy for computation    
            cls.pyFile.write('''\n
import math
import pickle
import pandas as pd
import numpy as np''')
            # In SAS Viya 4.0 and SAS Open Model Manager, a settings.py file is generated that points to the resource location
            if not isViya35:
                cls.pyFile.write('''\
import settings''')
                
            # Use a global variable for the model in order to load from memory only once
            cls.pyFile.write('''\n\n
global _thisModelFit''')
            
            # For H2O models, include the server initialization, or h2o.connect() call to use an H2O server
            if isH2OModel:
                cls.pyFile.write('''\n
h2o.init()''')

            # For each case of SAS Viya version and H2O model or not, load the model file as variable _thisModelFit
            if isViya35 and not isH2OModel:
                cls.pyFile.write(f'''\n
with open('/models/resources/viya/{modelID}/{modelFileName}', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)''')
            elif isViya35 and isH2OModel:
                cls.pyFile.write(f'''\n
with gzip.open('/models/resources/viya/{modelID}/{modelFileName}', 'r') as fileIn, open('/models/resources/viya/{modelID}/{modelFileName[:-4]}' + '.zip', 'wb') as fileOut:
    shutil.copyfileobj(fileIn, fileOut)
os.chmod('/models/resources/viya/{modelID}/{modelFileName[:-4]}' + '.zip', 0o777)
_thisModelFit = h2o.import_mojo('/models/resources/viya/{modelID}/{modelFileName[:-4]}' + '.zip')''')
            elif not isViya35 and not isH2OModel:
                cls.pyFile.write(f'''\n
with open(settings.pickle_path + '{modelFileName}', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)''')
            elif not isViya35 and isH2OModel:
                cls.pyFile.write(f'''\n
with gzip.open(settings.pickle_path + '{modelFileName}', 'r') as fileIn, open(settings.pickle_path + '{modelFileName[:-4]}' + '.zip', 'wb') as fileOut:
    shutil.copyfileobj(fileIn, fileOut)
os.chmod(settings.pickle_path + '{modelFileName[:-4]}' + '.zip', 0o777)
_thisModelFit = h2o.import_mojo(settings.pickle_path + '{modelFileName[:-4]}' + '.zip')''')
            
            # Create the score function with variables from the input dataframe provided and create the output variable line for SAS Model Manager
            cls.pyFile.write(f'''\n
def score{modelPrefix}({', '.join(inputVarList)}):
    "Output: {', '.join(metrics)}"''')
            # As a check for missing model variables, run a try/except block that reattempts to load the model in as a variable
            cls.pyFile.write(f'''\n
    try:
        _thisModelFit
    except NameError:\n''')
            if isViya35 and not isH2OModel:
                cls.pyFile.write(f'''
        with open('/models/resources/viya/{modelID}/{modelFileName}', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)''')
            elif isViya35 and isH2OModel:
                cls.pyFile.write(f'''
        _thisModelFit = h2o.import_mojo('/models/resources/viya/{modelID}/{modelFileName[:-4]}' + '.zip')''')
            elif not isViya35 and not isH2OModel:
                cls.pyFile.write(f'''
        with open(settings.pickle_path + '{modelFileName}', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)''')
            elif not isViya35 and isH2OModel:
                cls.pyFile.write(f'''
_thisModelFit = h2o.import_mojo(settings.pickle_path + '{modelFileName[:-4]}' + '.zip')''')
            
            if missingValues:
                # For each input variable, impute for missing values based on variable dtype
                for i, dTypes in enumerate(inputDtypesList):
                    dTypes = dTypes.name
                    if 'int' in dTypes or 'float' in dTypes:
                        if cls.checkIfBinary(inputDF[inputVarList[i]]):
                            cls.pyFile.write(f'''\n
    try:
        if math.isnan({inputVarList[i]}):
            {inputVarList[i]} = {float(list(inputDF[inputVarList[i]].mode())[0])}
    except TypeError:
        {inputVarList[i]} = {float(list(inputDF[inputVarList[i]].mode())[0])}''')
                        else:
                            cls.pyFile.write(f'''\n
    try:
        if math.isnan({inputVarList[i]}):
            {inputVarList[i]} = {float(inputDF[inputVarList[i]].mean(axis=0, skipna=True))}
    except TypeError:
        {inputVarList[i]} = {float(inputDF[inputVarList[i]].mean(axis=0, skipna=True))}''')
                    elif 'str' in dTypes or 'object' in dTypes:
                        cls.pyFile.write(f'''\n
    try:
        categoryStr = {inputVarList[i]}.strip()
    except AttributeError:
        categoryStr = 'Other'\n''')
                        tempVar = cls.splitStringColumn(inputDF[inputVarList[i]],
                                                         otherVariable)
                        newVarList.remove(inputVarList[i])
                        newVarList.extend(tempVar)
    
            # For non-H2O models, insert the model into the provided predictMethod call
            if not isH2OModel:
                predictMethod = predictMethod.format('_thisModelFit', 'inputArray')
                cls.pyFile.write(f'''\n
    try:
        inputArray = pd.DataFrame([[{', '.join(newVarList)}]],
                                  columns=[{', '.join(f"'{x}'" for x in newVarList)}],
                                  dtype=float)
        prediction = {predictMethod}
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required
    # For example, many statsmodels models include an intercept value that must be included for the model prediction
        inputArray = pd.DataFrame([[1.0, {', '.join(newVarList)}]],
                                columns=['const', {', '.join(f"'{x}'" for x in newVarList)}],
                                dtype=float)
        prediction = {predictMethod}''')
            elif isH2OModel:
                columnType = []
                for (var, dtype) in zip(newVarList, inputDtypesList):
                    if 'string' in dtype.name:
                        type = 'string'
                    else:
                        type = 'numeric'
                    columnType.append('\'' + var + '\'' + ':' + '\'' + type + '\'')
                cls.pyFile.write(f'''\n
    inputArray = pd.DataFrame([[{', '.join(newVarList)}]],
                              columns=[{', '.join(f"'{x}'" for x in newVarList)}],
                              dtype=float, index=[0])
    columnTypes = {{{', '.join(columnType)}}}
    h2oArray = h2o.H2OFrame(inputArray, column_types=columnTypes)
    prediction = _thisModelFit.predict(h2oArray)
    prediction = h2o.as_list(prediction, use_pandas=False)''')
            
            if not isH2OModel:
                cls.pyFile.write(f'''\n
    try:
        {metrics[0]} = float(prediction)
    except TypeError:
    # If the model expects non-binary responses, a TypeError will be raised.
    # The except block shifts the prediction to accept a non-binary response.
        {metrics[0]} = float(prediction[:,1])''')
                if threshPrediction is None:
                    threshPrediction = np.mean(targetDF)
                    cls.pyFile.write(f'''\n
    if ({metrics[0]} >= {threshPrediction}):
        {metrics[1]} = '1'
    else:
        {metrics[1]} = '0' ''')
            elif isH2OModel:
                cls.pyFile.write(f'''\n
    {metrics[0]} = float(prediction[1][2])
    {metrics[1]} = prediction[1][0]''')
            
            cls.pyFile.write(f'''\n
    return({metrics[0]}, {metrics[1]})''')
            
        if isViya35:
            with open(pyPath, 'r') as pFile:
                files = [dict(name=f'{modelPrefix}Score.py', file=pFile, role='score')]
                upload_and_copy_score_resources(modelID, files)
            modelRepo.convert_python_to_ds2(modelID)
            
    def splitStringColumn(self, inputSeries, otherVariable):
        '''
        Splits a column of string values into a number of new variables equal
        to the number of unique values in the original column (excluding None
        values). It then writes to a file the statements that tokenize the newly
        defined variables.
        
        Here is an example: Given a series named strCol with values ['A', 'B', 'C',
        None, 'A', 'B', 'A', 'D'], designates the following new variables:
        strCol_A, strCol_B, strCol_D. It then writes the following to the file:
            strCol_A = np.where(val == 'A', 1.0, 0.0)
            strCol_B = np.where(val == 'B', 1.0, 0.0)
            strCol_D = np.where(val == 'D', 1.0, 0.0)
                    
        Parameters
        ---------------
        inputSeries : string series
            Series with the string dtype.
        cls.pyFile : file (class variable)
            Open python file to write into.
            
        Returns
        ---------------
        newVarList : string list
            List of all new variable names split from unique values.
        '''
        
        uniqueValues = inputSeries.unique()
        uniqueValues = list(filter(None, uniqueValues))
        uniqueValues = [x for x in uniqueValues if str(x) != 'nan']
        newVarList = []
        for i, uniq in enumerate(uniqueValues):
            uniq = uniq.strip()
            if not uniq.isidentifier():
                raise SyntaxError('Invalid column value in inputDF. Values must be ' +
                                  'valid as Python variables (or easily space strippable).')
            newVarList.append(f'{inputSeries.name}_{uniq}')
            cls.pyFile.write(f'''
    {newVarList[i]} = np.where(categoryStr == '{uniq}', 1.0, 0.0)''')
                
        if ('Other' not in uniqueValues) and otherVariable:
            newVarList.append(f'{inputSeries.name}_Other')
            cls.pyFile.write(f'''
    {inputSeries.name}_Other = np.where(categoryStr == 'Other', 1.0, 0.0)''')
            
        return newVarList
    
    def checkIfBinary(self, inputSeries):
        '''
        Checks a pandas series to determine whether the values are binary or nominal.
        
        Parameters
        ---------------
        inputSeries : float or int series
            A series with numeric values.
        
        Returns
        ---------------
        isBinary : boolean
            The returned value is True if the series values are binary, and False if the series values
            are nominal.
        '''
        
        isBinary = False
        binaryFloat = [float(1), float(0)]
        
        if inputSeries.value_counts().size == 2:
            if (binaryFloat[0] in inputSeries.astype('float') and 
                binaryFloat[1] in inputSeries.astype('float')):
                isBinary = False
            else:
                isBinary = True
                
        return isBinary
        
