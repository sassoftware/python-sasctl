# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
from pathlib import Path
import sys

import swat
import getpass
import json
import pandas as pd
from sklearn import metrics
import numpy as np
from scipy.stats import kendalltau, gamma

# %%
class JSONFiles():
        
    def writeVarJSON(self, inputDF, isInput=True,
                     jPath=Path.cwd(), debug=False):
        '''
        Writes a variable descriptor JSON file for input or output variables,
        based on an input dataframe containing predictor and prediction columns.
        
        Parameters
        ---------------
        inputDF : Dataframe
            Input dataframe containing the training dataset in a 
            pandas.Dataframe format. Columns are used to define predictor and
            prediction variables (ambiguously named "predict").
        isInput : boolean
            Boolean to check if generating the input or output variable JSON.
        jPath : string, optional
            File location for the output JSON file. Default is the current
            working directory.
        debug : boolean, optional
            Debug mode to check predictor classification. The default is False.
            
        Yields
        ---------------
        {'inputVar.json', 'outputVar.json'}
            Output JSON file located at jPath.
        '''
        
        predictNames = inputDF.columns.values.tolist()
        outputJSON = pd.DataFrame()
        
        # loop through all predict variables to determine their name, length,
        # type, and level; append each to outputJSON
        for name in predictNames:
            predict = inputDF[name]
            firstRow = predict.loc[predict.first_valid_index()]
            dType = predict.dtypes.name
            dKind = predict.dtypes.kind
            isNum = pd.api.types.is_numeric_dtype(firstRow)
            isStr = pd.api.types.is_string_dtype(predict)
            
            # in debug mode, print each variables descriptor
            if debug:
                print('predictor = ', name)
                print('dType = ', dType)
                print('dKind = ', dKind)
                print('isNum = ', isNum)
                print('isStr = ', isStr)
                
            if isNum:
                if dType == 'category':
                    outputLevel = 'nominal'
                else:
                    outputLevel = 'interval'
                outputType = 'decimal'
                outputLength = 8
            elif isStr:
                outputLevel = 'nominal'
                outputType = 'string'
                outputLength = predict.str.len().max()
                
            outputRow = pd.Series([name,
                                   outputLength,
                                   outputType,
                                   outputLevel],
                                  index=['name',
                                         'length',
                                         'type',
                                         'level']
                                  )
            outputJSON = outputJSON.append([outputRow], ignore_index=True)
        
        if isInput:
            fileName = 'inputVar.json'
        else:
            fileName = 'outputVar.json'
            
        with open(Path(jPath) / fileName, 'w') as jFile:
            dfDump = pd.DataFrame.to_dict(outputJSON.transpose()).values()
            json.dump(list(dfDump),
                      jFile,
                      indent=4,
                      skipkeys=True)
            
    def writeModelPropertiesJSON(self, modelName, modelDesc, targetVariable,
                                 modelType, modelPredictors, targetEvent,
                                 numTargetCategories, eventProbVar=None,
                                 jPath=Path.cwd(), modeler=None):
        '''
        Writes a model properties JSON file. The JSON file format is required by the
        Model Repository API service and only eventProbVar can be 'None'.
        
        Parameters
        ---------------
        modelName : string
            User-defined model name.
        modelDesc : string
            User-defined model description.
        targetVariable : string
            Target variable to be predicted by the model.
        modelType : string
            User-defined model type.
        modelPredictors : string list
            List of predictor variables for the model.
        targetEvent : string
            Model target event (i.e. 1 for a binary event).
        numTargetCategories : int
            Number of possible target categories (i.e. 2 for a binary event).
        eventProbVar : string, optional
            Model prediction metric for scoring. Default is None.
        jPath : string, optional
            Location for the output JSON file. The default is the current
            working directory.
        modeler : string, optional
            The modeler name to be displayed in the model properties. The
            default value is None.
            
        Yields
        ---------------
        'ModelProperties.json'
            Output JSON file located at jPath.            
        '''
        
        description = modelDesc + ' : ' + targetVariable + ' = '
        # loop through all modelPredictors to write out the model description
        for counter, predictor in enumerate(modelPredictors):
            if counter > 0:
                description = description + ' + '
            description += predictor
            
        if numTargetCategories > 2:
            targetLevel = 'NOMINAL'
        else:
            targetLevel = 'BINARY'
            
        if eventProbVar is None:
            eventProbVar = 'P_' + targetVariable + targetEvent
        # Replace <myUserID> with the user ID of the modeler that created the model.
        if modeler is None:
            try:
                modeler = getpass.getuser()
            except OSError:
                modeler = '<myUserID>'
        
        pythonVersion = sys.version.split(' ', 1)[0]
        
        propIndex = ['name', 'description', 'function',
                     'scoreCodeType', 'trainTable', 'trainCodeType',
                     'algorithm', 'targetVariable', 'targetEvent',
                     'targetLevel', 'eventProbVar', 'modeler',
                     'tool', 'toolVersion']
        
        modelProperties = [modelName, description, 'classification',
                           'python', ' ', 'Python',
                           modelType, targetVariable, targetEvent,
                           targetLevel, eventProbVar, modeler,
                           'Python 3', pythonVersion]
        
        outputJSON = pd.Series(modelProperties, index=propIndex)
        
        with open(Path(jPath) / 'ModelProperties.json', 'w') as jFile:
            dfDump = pd.Series.to_dict(outputJSON.transpose())
            json.dump(dfDump,
                      jFile,
                      indent=4,
                      skipkeys=True)
            
    def writeFileMetadataJSON(self, modelPrefix, jPath=Path.cwd()):
        '''
        Writes a file metadata JSON file pointing to all relevant files.
        
        Parameters
        ---------------
        modelPrefix : string
            The variable for the model name that is used when naming model files.
            (i.e. hmeqClassTree + [Score.py || .pickle]).
        jPath : string, optional
            Location for the output JSON file. The default value is the current
            working directory.   
            
        Yields
        ---------------
        'fileMetadata.json'
            Output JSON file located at jPath.            
        '''
        
        fileMetadata = pd.DataFrame([['inputVariables', 'inputVar.json'],
                                     ['outputVariables', 'outputVar.json'],
                                     ['score', modelPrefix + 'Score.py'],
                                     ['python pickle',
                                      modelPrefix + '.pickle']],
                                    columns = ['role', 'name']
                                    )
        
        with open(Path(jPath) / 'fileMetadata.json', 'w') as jFile:
            dfDump = pd.DataFrame.to_dict(fileMetadata.transpose()).values()
            json.dump(list(dfDump),
                      jFile,
                      indent=4,
                      skipkeys=True)
            
    def writeBaseFitStat(self, csvPath=None, jPath=Path.cwd(),
                         userInput=False, tupleList=None):
        '''
        Writes a JSON file to display fit statistics for the model in SAS Open Model Manager.
        There are three modes to add fit parameters to the JSON file:
            
            1. Call the function with additional tuple arguments containing
            the name of the parameter, its value, and the partition it 
            belongs to.
            
            2. Provide line by line user input prompted by the function.
            
            3. Import values from a CSV file. Format should contain the above
            tuple in each row.
            
        The following are the base statistical parameters SAS Model Manager
        and SAS Open Model Manager currently supports:
            * RASE = Root Average Squared Error
            * NObs = Sum of Frequencies
            * GINI = Gini Coefficient
            * GAMMA = Gamma
            * MCE = Misclassification Rate
            * ASE = Average Squared Error
            * MCLL = Multi-Class Log Loss
            * KS = KS (Youden)
            * KSPostCutoff = ROC Separation
            * DIV = Divisor for ASE
            * TAU = Tau
            * KSCut = KS Cutoff
            * C = Area Under ROC
        
        Parameters
        ---------------
        csvPath : string, optional
            Location for an input CSV file that contains parameter statistics.
            The default value is None.
        jPath : string, optional
            Location for the output JSON file. The default value is the current
            working directory.
        userInput : boolean, optional
            If true, prompt the user for more parameters. The default value is false.
        tupleList : list of tuples, optional
            Input parameter tuples in the form of (parameterName, 
            parameterLabel, parameterValue, dataRole). For example,
            a sample parameter call would be ('NObs', 'Sum of Frequencies',
            3488, 'TRAIN'). Variable dataRole is typically either TRAIN,
            TEST, or VALIDATE or 1, 2, 3 respectively. The default value is None.
        
        Yields
        ---------------
        'dmcas_fitstat.json'
            Output JSON file located at jPath.
        '''
        
        validParams = ['_RASE_', '_NObs_', '_GINI_', '_GAMMA_', '_MCE_',
                       '_ASE_', '_MCLL_', '_KS_', '_KSPostCutoff_', '_DIV_',
                       '_TAU_', '_KSCut_', '_C_']
        
        nullJSONPath = Path(__file__).resolve().parent / 'null_dmcas_fitstat.json'
        nullJSONDict = self.readJSONFile(nullJSONPath)
        
        dataMap = [{}, {}, {}]
        for i in range(3):
            dataMap[i] = nullJSONDict['data'][i]
        
        if tupleList is not None:
            for paramTuple in tupleList:
                # ignore incorrectly formatted input arguments
                if type(paramTuple) == tuple and len(paramTuple) == 3:
                    paramName = self.formatParameter(paramTuple[0])
                    if paramName not in validParams:
                        continue
                    if type(paramTuple[2]) == str:
                        dataRole = self.convertDataRole(paramTuple[2]) 
                    else:
                        dataRole = paramTuple[2]
                    dataMap[dataRole-1]['dataMap'][paramName] = paramTuple[1]
        
        if userInput:        
            while True:
                paramName = input('Parameter name: ')
                paramName = self.formatParameter(paramName)
                if paramName not in validParams:
                    print('Not a valid parameter. Please see documentation.')
                    if input('More parameters? (Y/N)') == 'N':
                        break
                    continue
                paramValue = input('Parameter value: ')
                dataRole = input('Data role: ')
                
                if type(dataRole) is str:
                    dataRole = self.convertDataRole(dataRole)
                dataMap[dataRole-1]['dataMap'][paramName] = paramValue
                
                if input('More parameters? (Y/N)') == 'N':
                    break
        
        if csvPath is not None:
            csvData = pd.read_csv(csvPath)
            for i, row in enumerate(csvData.values):
                paramName, paramValue, dataRole = row
                paramName = self.formatParameter(paramName)
                if paramName not in validParams:
                    continue
                if type(dataRole) is str:
                    dataRole = self.convertDataRole(dataRole)
                dataMap[dataRole-1]['dataMap'][paramName] = paramValue
                
        outJSON = nullJSONDict
        for i in range(3):
            outJSON['data'][i] = dataMap[i]
                
        with open(Path(jPath) / 'dmcas_fitstat.json', 'w') as jFile:
            json.dump(outJSON,
                      jFile,
                      indent=4,
                      skipkeys=True)
            
    def calculateFitStat(self, validateData=None, trainData=None, 
                         testData=None, jPath=Path.cwd()):
        '''
        Calculates fit statistics from user data and predictions, then writes to
        a JSON file for importing into the common model repository. Note that if
        no dataset is provided (validate, train, or test), this function raises
        an error and does not create a JSON file.
        
        Datasets can be provided in the following forms:
        * pandas dataframe; the actual and predicted values are their own columns
        * numpy array; the actual and predicted values are their own columns or rows 
        and ordered such that the actual values come first and the predicted second
        * list; the actual and predicted values are their own indexed entry
        
        Parameters
        ---------------
        validateData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the validation dataset, including both
            the actual and predicted values. The default value is None.
        trainData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the train dataset, including both
            the actual and predicted values. The default value is None.
        testData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the test dataset, including both
            the actual and predicted values. The default value is None.
        jPath : string, optional
            Location for the output JSON file. The default value is the current
            working directory.
        
        Yields
        ---------------
        'dmcas_fitstat.json'
            Output JSON file located at jPath.
        '''
        
        nullJSONPath = Path(__file__).resolve().parent / 'null_dmcas_fitstat.json'
        nullJSONDict = self.readJSONFile(nullJSONPath)
        
        dataSets = [[[None], [None]],
                    [[None], [None]],
                    [[None], [None]]]
        
        dataPartitionExists = []
        for i, data in enumerate([validateData, trainData, testData]):
            if data is not None:
                dataPartitionExists.append(i)
                if type(data) is np.ndarray:
                    dataSets[i] = data.tolist()
                elif type(data) is pd.core.frame.DataFrame:
                    dataSets[i] = data.transpose().values.tolist()
                elif type(data) is list:
                    dataSets[i] = data
                            
        for j in dataPartitionExists:
            fitStats = nullJSONDict['data'][j]['dataMap']

            fitStats['_PartInd_'] = j
            
            fpr, tpr, _ = metrics.roc_curve(dataSets[j][0], dataSets[j][1])
        
            RASE = np.sqrt(metrics.mean_squared_error(dataSets[j][0], dataSets[j][1]))
            fitStats['_RASE_'] = RASE
        
            NObs = len(dataSets[j][0])
            fitStats['_NObs_'] = NObs
        
            auc = metrics.roc_auc_score(dataSets[j][0], dataSets[j][1])
            GINI = (2 * auc) - 1
            fitStats['_GINI_'] = GINI
        
            _, _, scale = gamma.fit(dataSets[j][1])
            fitStats['_GAMMA_'] = 1/scale
        
            intPredict = [round(x) for x in dataSets[j][1]]
            MCE = 1 - metrics.accuracy_score(dataSets[j][0], intPredict)
            fitStats['_MCE_'] = MCE
        
            ASE = metrics.mean_squared_error(dataSets[j][0], dataSets[j][1])
            fitStats['_ASE_'] = ASE
        
            MCLL = metrics.log_loss(dataSets[j][0], dataSets[j][1])
            fitStats['_MCLL_'] = MCLL
        
            KS = max(np.abs(fpr-tpr))
            fitStats['_KS_'] = KS
        
            KSPostCutoff = None
            fitStats['_KSPostCutoff_'] = KSPostCutoff
        
            DIV = len(dataSets[j][0])
            fitStats['_DIV_'] = DIV
        
            TAU, _ = kendalltau(dataSets[j][0], dataSets[j][1])
            fitStats['_TAU_'] = TAU
        
            KSCut = None
            fitStats['_KSCut_'] = KSCut
        
            C = metrics.auc(fpr, tpr)
            fitStats['_C_'] = C
        
            nullJSONDict['data'][j]['dataMap'] = fitStats

        with open(Path(jPath) / 'dmcas_fitstat.json', 'w') as jFile:
            json.dump(nullJSONDict, jFile, indent=4)            
    
    def generateROCLiftStat(self, targetName, targetValue, swatConn, 
                            validateData=None, trainData=None, testData=None, 
                            jPath=Path.cwd()):
        '''
        Calculates the ROC & Lift curves from user data and model predictions, then 
        writes it to JSON files for importing in to the common model repository.
        ROC & Lift calculations are completed by CAS through a SWAT call. Note that if
        no dataset is provided (validate, train, or test), this function raises
        an error and does not create any JSON files.
        
        Parameters
        ---------------
        targetName: str
            Target variable name to be predicted.
        targetValue: int or float
            Value of target variable that indicates an event.
        swatConn: SWAT connection to CAS
            Connection object to CAS service in SAS Model Manager or SAS Open 
            Model Manager through SWAT authentication.
        validateData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the validation dataset, including both
            the actual and predicted values. The default value is None.
        trainData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the train dataset, including both
            the actual and predicted values. The default value is None.
        testData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the test dataset, including both
            the actual and predicted values. The default value is None.
        jPath : string, optional
            Location for the output JSON file. The default value is the current
            working directory.
        
        Yields
        ---------------
        'dmcas_roc.json'
            Output JSON file located at jPath.
        'dmcas_lift.json'
            Output JSON file located at jPath.
        '''
        
        nullJSONROCPath = Path(__file__).resolve().parent / 'null_dmcas_roc.json'
        nullJSONROCDict = self.readJSONFile(nullJSONROCPath)
        
        nullJSONLiftPath = Path(__file__).resolve().parent / 'null_dmcas_lift.json'
        nullJSONLiftDict = self.readJSONFile(nullJSONLiftPath)
                
        dataSets = pd.DataFrame()

        dataPartitionExists = []
        # Check if a data partition exists, then convert to a pandas dataframe
        for i, data in enumerate([validateData, trainData, testData]):
            if data is not None:
                dataPartitionExists.append(i)
                columns = [f'actual{i}', f'predicted{i}']
                if type(data) is np.ndarray:
                    try:
                        dataSets[columns] = data
                    except ValueError:
                        dataSets[columns] = data.transpose()
                elif type(data) is list:
                    dataSets[columns] = np.array(data).transpose()
                elif type(data) is pd.core.frame.DataFrame:
                    dataSets[columns] = data.values
                    
        if len(dataPartitionExists) == 0:
            try:
                raise ValueError
            except ValueError:
                print('No data was provided. Please provide the actual ' +
                      'and predicted values for at least one of the ' + 
                      'partitions (VALIDATE, TRAIN, or TEST).')
                raise
        
        swatConn.loadactionset('percentile')
        
        for i in dataPartitionExists:
            columns = [f'actual{i}', f'predicted{i}']
            predicted = f'predicted{i}'
            swatConn.read_frame(dataSets[columns],
                                casout=dict(name='scoredValues',
                                            replace=True))
            swatConn.percentile.assess(table='scoredValues',
                                       inputs=predicted,
                                       casout=dict(name='scoreAssess',
                                                   replace=True),
                                       response=targetName,
                                       event=targetValue)
            assessROC = swatConn.CASTable('scoreAssess_ROC').to_frame()
            assessLift = swatConn.CASTable('scoreAssess').to_frame()

            for j in range(100):
                rowNumber = (i*100) + j
                nullJSONROCDict['data'][rowNumber]['dataMap']['_Event_'] = targetValue
                nullJSONROCDict['data'][rowNumber]['dataMap']['_TargetName_'] = targetName
                nullJSONROCDict['data'][rowNumber]['dataMap']['_Cutoff_'] = assessROC['_Cutoff_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_TP_'] = assessROC['_TP_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_FP_'] = assessROC['_FP_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_FN_'] = assessROC['_FN_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_TN_'] = assessROC['_TN_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_Sensitivity_'] = assessROC['_Sensitivity_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_Specificity_'] = assessROC['_Specificity_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_KS_'] = assessROC['_KS_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_KS2_'] = assessROC['_KS2_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_FHALF_'] = assessROC['_FHALF_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_FPR_'] = assessROC['_FPR_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_ACC_'] = assessROC['_ACC_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_FDR_'] = assessROC['_FDR_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_F1_'] = assessROC['_F1_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_C_'] = assessROC['_C_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_GINI_'] = assessROC['_GINI_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_GAMMA_'] = assessROC['_GAMMA_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_TAU_'] = assessROC['_TAU_']
                nullJSONROCDict['data'][rowNumber]['dataMap']['_MiscEvent_'] = assessROC['_MiscEvent_']
            
            for j in range(21):
                rowNumber = (i*21) + j
                nullJSONLiftDict['data'][rowNumber]['dataMap']['_Event_'] = targetValue
                nullJSONLiftDict['data'][rowNumber]['dataMap']['_TargetName_'] = targetName
                if j != 0:
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_Depth_'] = assessLift['_Depth_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_Value_'] = assessLift['_Value_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_NObs_'] = assessLift['_NObs_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_NEvents_'] = assessLift['_NEvents_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_NEventsBest_'] = assessLift['_NEventsBest_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_Resp_'] = assessLift['_Resp_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_RespBest_'] = assessLift['_RespBest_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_Lift_'] = assessLift['_Lift_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_LiftBest_'] = assessLift['_LiftBest_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_CumResp_'] = assessLift['_CumResp_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_CumRespBest_'] = assessLift['_CumRespBest_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_CumLift_'] = assessLift['_CumLift_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_CumLiftBest_'] = assessLift['_CumLiftBest_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_PctResp_'] = assessLift['_PctResp_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_PctRespBest_'] = assessLift['_PctRespBest_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_CumPctResp_'] = assessLift['_CumPctResp_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_CumPctRespBest_'] = assessLift['_CumPctRespBest_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_Gain_'] = assessLift['_Gain_']
                    nullJSONLiftDict['data'][rowNumber]['dataMap']['_GainBest_'] = assessLift['_GainBest_']
        
        with open(Path(jPath) / 'dmcas_roc.json', 'w') as jFile:
            json.dump(nullJSONROCDict, jFile, indent=4) 
            
        with open(Path(jPath) / 'dmcas_lift.json', 'w') as jFile:
            json.dump(nullJSONLiftDict, jFile, indent=4)
                        
    def readJSONFile(self, path):
        '''
        Reads a JSON file from a given path.
        
        Parameters
        ----------
        path : str or pathlib Path
            Location of the JSON file to be opened.
            
        Returns
        -------
        json.load(jFile) : str
            String contents of json file.
        '''
        
        with open(path) as jFile:
            return(json.load(jFile))
            
    def formatParameter(self, paramName):
        '''
        Formats the parameter name to the JSON standard (_<>_). No changes are
        applied if the string is already formatted correctly.
        
        Parameters
        ---------------
        paramName : string
            Name of the parameter.
            
        Returns
        ---------------
        paramName : string
            Name of the parameter.
        '''
        
        if not (paramName.startswith('_') and paramName.endswith('_')):
            if not paramName.startswith('_'):
                paramName = '_' + paramName
            if not paramName.endswith('_'):
                paramName = paramName + '_'
        
        return paramName
            
    def convertDataRole(self, dataRole):
        '''
        Converts the data role identifier from string to int or int to string. JSON
        file descriptors require the string, int, and formatted int. If the
        provided data role is not valid, defaults to TRAIN (1).
        
        Parameters
        ---------------
        dataRole : string or int
            Identifier of the dataset's role; either TRAIN, TEST, or VALIDATE,
            which correspond to 1, 2, or 3.
            
        Returns
        ---------------
        conversion : int or string
            Converted data role identifier.
        '''
        
        if type(dataRole) is int or type(dataRole) is float:
            dataRole = int(dataRole)
            if dataRole == 1:
                conversion = 'TRAIN'
            elif dataRole == 2:
                conversion = 'TEST'
            elif dataRole == 3:
                conversion = 'VALIDATE'
            else:
                conversion = 'TRAIN'
        elif type(dataRole) is str:
            if dataRole == 'TRAIN':
                conversion = 1
            elif dataRole == 'TEST':
                conversion = 2    
            elif dataRole == 'VALIDATE':
                conversion = 3
            else:
                conversion = 1
        else:
            conversion = 1
                
        return conversion