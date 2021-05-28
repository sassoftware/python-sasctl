# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import numpy as np
import re
from ..core import platform_version
from .._services.model_repository import ModelRepository as modelRepo

# %%
class ScoreCode():
    
    @classmethod
    def writeScoreCode(cls, inputDF, targetDF, modelPrefix,
                       predictMethod, modelFileName,
                       metrics=['EM_EVENTPROBABILITY', 'EM_CLASSIFICATION'],
                       pyPath=Path.cwd(), threshPrediction=None,
                       otherVariable=False, model=None, isH2OModel=False, missingValues=False,
                       scoreCAS=True):
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
        scoreCAS : boolean, optional
    		Sets whether models registered to SAS Viya 3.5 should be able to be scored and
            validated through both CAS and SAS Micro Analytic Service. By default true. If
            set to false, then the model will only be able to be scored and validated through
            SAS Micro Analytic Service. By default true.

    	Yields
    	------
        '*Score.py'
            The Python score code file for the model.
        'dcmas_epscorecode.sas' (for SAS Viya 3.5 models)
            Python score code wrapped in DS2 and prepared for CAS scoring or publishing.
        'dmcas_packagescorecode.sas' (for SAS Viya 3.5 models)
            Python score code wrapped in DS2 and prepared for SAS Microanalyic Service scoring or publishing.
        '''       
        # Call REST API to check SAS Viya version
        isViya35 = (platform_version() == '3.5')
        
        # Initialize modelID to remove unbound variable warnings
        modelID = None

        # Helper method for uploading & migrating files
        # TODO: Integrate with register_model() or publish_model() task.
        def upload_and_copy_score_resources(model, files):
            for file in files:
                modelRepo.add_model_content(model, **file)
            return modelRepo.copy_python_resources(model)

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
        zPath = Path(pyPath)
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
                cls.pyFile.write('''\n
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
                cls.pyFile.write('''\n
with gzip.open('/models/resources/viya/{modelID}/{modelFileName}', 'r') as fileIn, open('/models/resources/viya/{modelID}/{modelZipFileName}', 'wb') as fileOut:
    shutil.copyfileobj(fileIn, fileOut)
os.chmod('/models/resources/viya/{modelID}/{modelZipFileName}', 0o777)
_thisModelFit = h2o.import_mojo('/models/resources/viya/{modelID}/{modelZipFileName}')'''.format(
                    modelID=modelID,
                    modelFileName=modelFileName,
                    modelZipFileName=modelFileName[:-4] + 'zip'
                ))
            elif not isViya35 and not isH2OModel:
                cls.pyFile.write('''\n
with open(settings.pickle_path + '{modelFileName}', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)'''.format(modelFileName=modelFileName))
            elif not isViya35 and isH2OModel:
                cls.pyFile.write('''\n
with gzip.open(settings.pickle_path + '{modelFileName}', 'r') as fileIn, open(settings.pickle_path + '{modelZipFileName}', 'wb') as fileOut:
    shutil.copyfileobj(fileIn, fileOut)
os.chmod(settings.pickle_path + '{modelZipFileName}', 0o777)
_thisModelFit = h2o.import_mojo(settings.pickle_path + '{modelZipFileName}')'''.format(modelFileName=modelFileName,
                                                                                                 modelZipFileName=modelFileName[:-4] + 'zip'
                                                                                                 ))
            # Create the score function with variables from the input dataframe provided and create the output variable line for SAS Model Manager
            cls.pyFile.write('''\n
def score{modelPrefix}({inputVarList}):
    "Output: {metrics}"'''.format(modelPrefix=modelPrefix,
                                             inputVarList=', '.join(inputVarList),
                                             metrics=', '.join(metrics)))
            # As a check for missing model variables, run a try/except block that reattempts to load the model in as a variable
            cls.pyFile.write('''\n
    try:
        _thisModelFit
    except NameError:\n''')
            if isViya35 and not isH2OModel:
                cls.pyFile.write('''
        with open('/models/resources/viya/{modelID}/{modelFileName}', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)'''.format(modelID=modelID, modelFileName=modelFileName))
            elif isViya35 and isH2OModel:
                cls.pyFile.write('''
        _thisModelFit = h2o.import_mojo('/models/resources/viya/{modelID}/{modelZipFileName}')
        '''.format(modelID=modelID,
                   modelZipFileName=modelFileName[:-4] + 'zip'))

            elif not isViya35 and not isH2OModel:
                cls.pyFile.write('''
        with open(settings.pickle_path + '{modelFileName}', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)'''.format(modelFileName=modelFileName))
            elif not isViya35 and isH2OModel:
                cls.pyFile.write('''
        _thisModelFit = h2o.import_mojo(settings.pickle_path + '{}')'''.format(modelFileName[:-4] + 'zip'))
            
            if missingValues:
                # For each input variable, impute for missing values based on variable dtype
                for i, dTypes in enumerate(inputDtypesList):
                    dTypes = dTypes.name
                    if 'int' in dTypes or 'float' in dTypes:
                        if cls.checkIfBinary(inputDF[inputVarList[i]]):
                            cls.pyFile.write('''\n
    try:
        if math.isnan({inputVar}):
            {inputVar} = {inputVarMode}
    except TypeError:
        {inputVar} = {inputVarMode}'''.format(inputVar=inputVarList[i],
                                             inputVarMode=float(list(inputDF[inputVarList[i]].mode())[0])))
                        else:
                            cls.pyFile.write('''\n
    try:
        if math.isnan({inputVar}):
            {inputVar} = {inputVarMean}
    except TypeError:
        {inputVar} = {inputVarMean}'''.format(inputVar=inputVarList[i],
                                              inputVarMean=float(inputDF[inputVarList[i]].mean(axis=0, skipna=True))))
                    elif 'str' in dTypes or 'object' in dTypes:
                        cls.pyFile.write('''\n
    try:
        categoryStr = {inputVar}.strip()
    except AttributeError:
        categoryStr = 'Other'\n'''.format(inputVar=inputVarList[i]))

                        tempVar = cls.splitStringColumn(inputDF[inputVarList[i]],
                                                         otherVariable)
                        newVarList.remove(inputVarList[i])
                        newVarList.extend(tempVar)
    
            # For non-H2O models, insert the model into the provided predictMethod call
            if not isH2OModel:
                predictMethod = predictMethod.format('_thisModelFit', 'inputArray')
                cls.pyFile.write('''\n
    try:
        inputArray = pd.DataFrame([[{newVars}]],
                                  columns=[{columns}],
                                  dtype=float)
        prediction = {predictMethod}
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required
    # For example, many statsmodels models include an intercept value that must be included for the model prediction
        inputArray = pd.DataFrame([[1.0, {newVars}]],
                                columns=['const', {columns}],
                                dtype=float)
        prediction = {predictMethod}'''.format(newVars=', '.join(newVarList),
                                               columns=', '.join("'%s'" % x for x in newVarList),
                                               predictMethod=predictMethod)),
            elif isH2OModel:
                columnType = []
                for (var, dtype) in zip(newVarList, inputDtypesList):
                    if 'string' in dtype.name:
                        type = 'string'
                    else:
                        type = 'numeric'
                    columnType.append('\'' + var + '\'' + ':' + '\'' + type + '\'')
                cls.pyFile.write('''\n
    inputArray = pd.DataFrame([[{newVars}]],
                              columns=[{columns}],
                              dtype=float, index=[0])
    columnTypes = {{{columnTypes}}}
    h2oArray = h2o.H2OFrame(inputArray, column_types=columnTypes)
    prediction = _thisModelFit.predict(h2oArray)
    prediction = h2o.as_list(prediction, use_pandas=False)'''.format(newVars=', '.join(newVarList),
                                                                     columns=', '.join("'%s'" % x for x in newVarList),
                                                                     columnTypes=', '.join(columnType)
                                                                     ))
            if not isH2OModel:
                cls.pyFile.write('''\n
    try:
        {metric} = float(prediction)
    except TypeError:
    # If the model expects non-binary responses, a TypeError will be raised.
    # The except block shifts the prediction to accept a non-binary response.
        {metric} = float(prediction[:,1])'''.format(metric=metrics[0]))
                if threshPrediction is None:
                    threshPrediction = np.mean(targetDF)
                    cls.pyFile.write('''\n
    if ({metric0} >= {threshold}):
        {metric1} = '1'
    else:
        {metric1} = '0' '''.format(metric0=metrics[0],
                                   metric1=metrics[1],
                                   threshold=threshPrediction))
            elif isH2OModel:
                cls.pyFile.write('''\n
    {} = float(prediction[1][2])
    {} = prediction[1][0]'''.format(metrics[0], metrics[1]))
            
            cls.pyFile.write('''\n
    return({}, {})'''.format(metrics[0], metrics[1]))
        
        # For SAS Viya 3.5, the model is first registered to SAS Model Manager, then the model UUID can be
        # added to the score code and reuploaded to the model file contents
        if isViya35:
            with open(pyPath, 'r') as pFile:
                files = [dict(name='{}Score.py'.format(modelPrefix), file=pFile, role='score')]
                upload_and_copy_score_resources(modelID, files)
            # After uploading the score code and migrating score resources, call the wrapper API to create
            # the Python score code wrapped in DS2
            modelRepo.convert_python_to_ds2(modelID)
            # Convert the DS2 wrapper code into two separate files: dmcas_epscorecode.sas and dmcas_packagescorecode.sas
            # The former scores or validates in CAS and the latter in SAS Microanalytic Service
            if scoreCAS:
                fileContents = modelRepo.get_model_contents(modelID)
                for item in fileContents:
                    if item.name == 'score.sas':
                        masCode = modelRepo.get('models/%s/contents/%s/content' % (item.modelId, item.id))
                with open(zPath / 'dmcas_packagescorecode.sas', 'w') as file:
                    print(masCode, file=file)
                casCode = cls.convertMAStoCAS(masCode, modelID)
                with open(zPath / 'dmcas_epscorecode.sas', 'w') as file:
                    print(casCode, file=file)
                for scoreCode in ['dmcas_packagescorecode.sas', 'dmcas_epscorecode.sas']:
                    with open(zPath / scoreCode, 'r') as file:
                        if scoreCode == 'dmcas_epscorecode.sas':
                            upload_and_copy_score_resources(modelID, [dict(name=scoreCode, file=file, role='score')])
                        else:
                            upload_and_copy_score_resources(modelID, [dict(name=scoreCode, file=file)])
                model = modelRepo.get_model(modelID)
                model['scoreCodeType'] = 'ds2MultiType'
                modelRepo.update_model(model)
            
    def splitStringColumn(cls, inputSeries, otherVariable):
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
            newVarList.append('{}_{}'.format(inputSeries.name, uniq))
            cls.pyFile.write('''
    {0} = np.where(categoryStr == '{1}', 1.0, 0.0)'''.format(newVarList[i], uniq))
                
        if ('Other' not in uniqueValues) and otherVariable:
            newVarList.append('{}_Other'.format(inputSeries.name))
            cls.pyFile.write('''
    {}_Other = np.where(categoryStr == 'Other', 1.0, 0.0)'''.format(inputSeries.name))
            
        return newVarList
    
    def checkIfBinary(inputSeries):
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
        
    def convertMAStoCAS(MASCode, modelId):
        '''Using the generated score.sas code from the Python wrapper API, 
        convert the SAS Microanalytic Service based code to CAS compatible.

        Parameters
        ----------
        MASCode : str
            String representation of the packagescore.sas DS2 wrapper 
        modelId : str or dict
            The name or id of the model, or a dictionary representation of
            the model

        Returns
        -------
        CASCode : str
            String representation of the epscorecode.sas DS2 wrapper code
        '''
        model = modelRepo.get_model(modelId)
        outputString = ''
        for outVar in model['outputVariables']:
            outputString = outputString + 'dcl '
            if outVar['type'] == 'string':
                outputString = outputString + 'varchar(100) '
            else:
                outputString = outputString + 'double '
            outputString = outputString  + outVar['name'] + ';\n'
        start = MASCode.find('score(')
        finish = MASCode[start:].find(');')
        scoreVars = MASCode[start+6:start+finish]
        inputString = ' '.join([x for x in scoreVars.split(' ') if (x != 'double' and x != 'in_out' and x != 'varchar(100)')])
        endBlock = 'method run();\n    set SASEP.IN;\n    score({});\nend;\nenddata;'.format(inputString)
        replaceStrings = {'package pythonScore / overwrite=yes;': 'data sasep.out;',
                          'dcl int resultCode revision;': 'dcl double resultCode revision;\n' + outputString,
                          'endpackage;': endBlock}
        replaceStrings = dict((re.escape(k), v) for k, v in replaceStrings.items())
        pattern = re.compile('|'.join(replaceStrings.keys()))
        casCode = pattern.sub(lambda m: replaceStrings[re.escape(m.group(0))], MASCode)
        return casCode