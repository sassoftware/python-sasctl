# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import numpy as np
import json
from uuid import uuid4
from ..tasks import get_software_version, upload_and_copy_score_resources
from .._services.model_repository import ModelRepository as modelRepo

# %%
class ScoreCode():
    
    def writeScoreCode(self, inputDF, targetDF, modelPrefix,
                       predictMethod, pickleName,
                       metrics=['EM_EVENTPROBABILITY', 'EM_CLASSIFICATION'],
                       pyPath=Path.cwd(), threshPrediction=None,
                       otherVariable=False, model=None):
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
        ---------------
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
        pickleName : string
            Name of the pickle file that contains the model.
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
    			
    	Yields
    	---------------
        '*Score.py'
            The Python score code file for the model.
        '''       
        if modelRepo.is_uuid(model):
            modelID = model
        elif isinstance(model, dict) and 'id' in model:
            modelID = model['id']
        else:
            model = modelRepo.get_model(model)
            modelID = model['id']
        
        isViya35 = (get_software_version() == '3.5')
        if isViya35 and (modelID == None):
            raise ValueError('The model UUID is required for score code written for' +
                             ' SAS Model Manager on SAS Viya 3.5.')
        
        inputVarList = list(inputDF.columns)
        for name in inputVarList:
            if not str(name).isidentifier():
                raise SyntaxError('Invalid column name in inputDF. Columns must be ' +
                                  'valid as Python variables.')
        newVarList = list(inputVarList)
        inputDtypesList = list(inputDF.dtypes)        
    
        zPath = pyPath
        pyPath = Path(pyPath) / (modelPrefix + 'Score.py')
        with open(pyPath, 'w') as self.pyFile:
        
            self.pyFile.write('''\
import math
import pickle
import pandas as pd
import numpy as np
import settings''')

            if isViya35:
                self.pyFile.write(f'''\n
with open('/models/resources/viya/{modelID}/{pickleName}', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)''')
            else:
                self.pyFile.write(f'''\n
with open(settings.pickle_path + '{pickleName}', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)''')
            
            self.pyFile.write(f'''\n
def score{modelPrefix}({', '.join(inputVarList)}):
    "Output: {', '.join(metrics)}"''')
            if isViya35:
                self.pyFile.write(f'''\n
    try:
        _thisModelFit
    except NameError:
        with open('/models/resources/viya/{modelID}/{pickleName}', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)''')
            else:
                self.pyFile.write(f'''\n
    try:
        _thisModelFit
    except NameError:
        with open(settings.pickle_path + '{pickleName}', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)''')
            
            for i, dTypes in enumerate(inputDtypesList):
                dTypes = dTypes.name
                if 'int' in dTypes or 'float' in dTypes:
                    if self.checkIfBinary(inputDF[inputVarList[i]]):
                        self.pyFile.write(f'''\n
    try:
        if math.isnan({inputVarList[i]}):
            {inputVarList[i]} = {float(list(inputDF[inputVarList[i]].mode())[0])}
    except TypeError:
        {inputVarList[i]} = {float(list(inputDF[inputVarList[i]].mode())[0])}''')
                    else:
                        self.pyFile.write(f'''\n
    try:
        if math.isnan({inputVarList[i]}):
            {inputVarList[i]} = {float(inputDF[inputVarList[i]].mean(axis=0, skipna=True))}
    except TypeError:
        {inputVarList[i]} = {float(inputDF[inputVarList[i]].mean(axis=0, skipna=True))}''')
                elif 'str' in dTypes or 'object' in dTypes:
                    self.pyFile.write(f'''\n
    try:
        categoryStr = {inputVarList[i]}.strip()
    except AttributeError:
        categoryStr = 'Other'\n''')
                    tempVar = self.splitStringColumn(inputDF[inputVarList[i]],
                                                     otherVariable)
                    newVarList.remove(inputVarList[i])
                    newVarList.extend(tempVar)
    
            # Insert the model into the provided predictMethod call.
            predictMethod = predictMethod.format('_thisModelFit', 'inputArray')
            self.pyFile.write(f'''\n
    try:
        inputArray = pd.DataFrame([[{', '.join(newVarList)}]],
                                columns = [{', '.join(f"'{x}'" for x in newVarList)}],
                                dtype = float)
        prediction = {predictMethod}
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required.
    # For example, many statsmodels models include an intercept value that must be included for the model prediction.
        inputArray = pd.DataFrame([[1.0, {', '.join(newVarList)}]],
                                columns = ['const', {', '.join(f"'{x}'" for x in newVarList)}],
                                dtype = float)
        prediction = {predictMethod}''')
            
            self.pyFile.write(f'''\n
    try:
        {metrics[0]} = float(prediction)
    except TypeError:
    # If the model expects non-binary responses, a TypeError will be raised.
    # The except block shifts the prediction to accept a non-binary response.
        {metrics[0]} = float(prediction[:,1])''')
            if threshPrediction is None:
                threshPrediction = np.mean(targetDF)
                self.pyFile.write(f'''\n
    if ({metrics[0]} >= {threshPrediction}):
        {metrics[1]} = '1'
    else:
        {metrics[1]} = '0' ''')
            
            self.pyFile.write(f'''\n
    return({metrics[0]}, {metrics[1]})''')
            
        if isViya35:
            self.convertPythonModeltoDS2(Path(zPath) / ('score.sas'), 
                                         zPath, pyPath, inputVarList,
                                         inputDtypesList, modelPrefix)
            files = [dict(name=f'{modelPrefix}Score.py', file=pyPath),
                     dict(name='score.sas', file=Path(zPath) / ('score.sas'),
                          role='Score code')]
            upload_and_copy_score_resources(modelID, files)
                
    def convertPythonModeltoDS2(self, sasPath, zPath, pyPath, inputVarList, inputDtypesList, modelPrefix):
        
        pythonCode = []
        with open(pyPath, 'r') as file:
            for line in file:
                pythonCode.append(line)
        
        with open(sasPath, 'w') as sasFile:
            sasFile.write('''
package pythonScore / overwrite=yes;
dcl package pymas pm;
dcl package logger logr('App.tk.MAS');
dcl varchar(32767) character set utf8 pypgm;
dcl int resultCode revision;\n''')
            
            # Separate input variables between string [varchar(100) in DS2] and not-string [double in DS2],
            # recombine into a single string, while maintaining argument order from the Python function
            index = [ind for ind, val in enumerate(inputDtypesList) if val != 'string']
            for ind, string in enumerate(inputVarList):
                if ind in index:
                    inputVarList[ind] = 'double ' + string
                else:
                    inputVarList[ind] = 'varchar(100) ' + string
            methodInputs = ', '.join(inputVarList)
            resultVar = 'in_out double resultCode'
            with open(zPath + 'outputVar.json', 'r') as file:
                outputJSON = json.load(file)
            methodOutputs = []
            for var in outputJSON:
                if var['type'] == 'string':
                    methodOutputs.append('in_out varchar(100) ' + var['name'])
                else:
                    methodOutputs.append('in_out double ' + var['name'])
            methodOutputs = ', '.join(methodOutputs)

            sasFile.write(f'''\n
method score({', '.join([methodInputs, resultVar, methodOutputs])});''')
            
            execUUID = 'model_exec_' + str(uuid4())
            sasFile.write(f'''\n
    resultCode = revision = 0;
    if null(pm) then do;
        pm = _new_ pymas();
        resultCode = pm.useModule('{execUUID}', 1);
        if resultCode then do;''')
            
            for line in pythonCode:
                sasFile.write(f'''\n
            ResultCode = pm.appendSrcLine({line})''')
            
            sasFile.write(f'''\n
            revision = pm.publish(pm.getSource(), '{execUUID}');\n\n
            if (revision < 1) then do;
                logr.log('e', 'py.publish() failed.');
                resultCode = -1;
                return;
            end;
        end;
    end;
    resultCode = pm.useMethod('{'score' + modelPrefix}');
    if resultCode then do;
        logr.log('E', 'useMethod() failed. resultCode=$s', resultCode);
        return;
    end;''')
            
            for ind, var in enumerate(inputVarList):
                if ind in index:
                    sasFile.write(f'''\n
    resultCode = pm.setDouble('{var}', {var});
    if resultCode then
        logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);''')
                else:
                    sasFile.write(f'''\n
    resultCode = pm.setString('{var}', {var});
    if resultCode then
        logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);''')

            sasFile.write('''\n
    resultCode = pm.execute();
    if (resultCode) then put 'Error: pm.execute failed.  resultCode=' resultCode;
    else do;''')
            for var in outputJSON:
                if var['type'] == 'string':
                    sasFile.write(f'''\n
        {var['name']} = pm.getString('{var['name']}');''')
                else:
                    sasFile.write(f'''\n
        {var['name']} = pm.getDouble('{var['name']}');''')
            sasFile.write('''\n
    end;
end;

endpackage;''')
            
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
        self.pyFile : file (class variable)
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
            self.pyFile.write(f'''
    {newVarList[i]} = np.where(categoryStr == '{uniq}', 1.0, 0.0)''')
                
        if ('Other' not in uniqueValues) and otherVariable:
            newVarList.append(f'{inputSeries.name}_Other')
            self.pyFile.write(f'''
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
        