# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
from pathlib import Path
import sys

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
            
        The following are the base statistical parameters SAS Open Model Manager
        currently supports:
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
            
    def calculateFitStat(self, data, jPath=Path.cwd()):
        '''
        Calculates fit statistics from user data and writes it to a JSON file for
        importing into the common model repository. Input data takes the form of a list of
        arrays, with the actual and predicted data separated into the
        following parititions: validate, train, test. An example is shown
        below:
            
            data = [(yValidateActual, yValidatePrediction),
                    (yTrainActual, yTrainPrediction),
                    (yTestActual, yTestPrediction)]
        
        For partitions without data, replace the array with a None assignment.
        
        Parameters
        ---------------
        data : list of arrays
            List of data arrays, separated into actual and predicted values
            and partitioned into Validate, Test, and Train sets. See above for 
            a formatted example.
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
        
        dataPartitionExists = []
        for i in range(3):
            if data[i][0] is not None:
                dataPartitionExists.append(i)
                
        for j in dataPartitionExists:
            fitStats = nullJSONDict['data'][j]['dataMap']

            fitStats['_PartInd_'] = j
            
            fpr, tpr, _ = metrics.roc_curve(data[j][0], data[j][1])
        
            RASE = np.sqrt(metrics.mean_squared_error(data[j][0], data[j][1]))
            fitStats['_RASE_'] = RASE
        
            NObs = len(data[j][0])
            fitStats['_NObs_'] = NObs
        
            auc = metrics.roc_auc_score(data[j][0], data[j][1])
            GINI = (2 * auc) - 1
            fitStats['_GINI_'] = GINI
        
            _, _, scale = gamma.fit(data[j][1])
            fitStats['_GAMMA_'] = 1/scale
        
            intPredict = [round(x) for x in data[j][1]]
            MCE = 1 - metrics.accuracy_score(data[j][0], intPredict)
            fitStats['_MCE_'] = MCE
        
            ASE = metrics.mean_squared_error(data[j][0], data[j][1])
            fitStats['_ASE_'] = ASE
        
            MCLL = metrics.log_loss(data[j][0], data[j][1])
            fitStats['_MCLL_'] = MCLL
        
            KS = max(np.abs(fpr-tpr))
            fitStats['_KS_'] = KS
        
            KSPostCutoff = None
            fitStats['_KSPostCutoff_'] = KSPostCutoff
        
            DIV = len(data[j][0])
            fitStats['_DIV_'] = DIV
        
            TAU, _ = kendalltau(data[j][0], data[j][1])
            fitStats['_TAU_'] = TAU
        
            KSCut = None
            fitStats['_KSCut_'] = KSCut
        
            C = metrics.auc(fpr, tpr)
            fitStats['_C_'] = C
        
            nullJSONDict['data'][j]['dataMap'] = fitStats

        with open(Path(jPath) / 'dmcas_fitstat.json', 'w') as jFile:
            json.dump(nullJSONDict, jFile, indent=4)            
    
    def generateROCStat(self, data, targetName, jPath=Path.cwd()):
        '''
        Calculates the ROC curve from user data and writes it to a JSON file for
        importing in to common model repository. Input data takes the form of a list of
        arrays, with the actual and predicted data separated into the
        following parititions: validate, train, test. An example is shown
        below:
            
            data = [(yValidateActual, yValidatePrediction),
                    (yTrainActual, yTrainPrediction),
                    (yTestActual, yTestPrediction)]
        
        For partitions without data, replace the array with a None assignment.
        
        Parameters
        ---------------
        data : list of arrays
            List of data arrays, separated into actual and predicted values
            and partitioned into Validate, Test, and Train sets. See above for 
            a formatted example.
        targetName: str
            Target variable name to be predicted.
        jPath : string, optional
            Location for the output JSON file. The default value is the current
            working directory.
        
        Yields
        ---------------
        'dmcas_roc.json'
            Output JSON file located at jPath.
        '''
        
        dictDataRole = {'parameter': '_DataRole_', 'type': 'char',
                        'label': 'Data Role', 'length': 10,
                        'order': 1, 'values': ['_DataRole_'],
                        'preformatted': False}

        dictPartInd = {'parameter': '_PartInd_', 'type': 'num',
                       'label': 'Partition Indicator', 'length': 8,
                       'order': 2, 'values': ['_PartInd_'],
                       'preformatted': False}
        
        dictPartIndf = {'parameter': '_PartInd__f', 'type': 'char',
                        'label': 'Formatted Partition', 'length': 12,
                        'order': 3, 'values': ['_PartInd__f'],
                        'preformatted': False}
        
        dictColumn = {'parameter': '_Column_', 'type': 'num',
                      'label': 'Analysis Variable', 'length': 32,
                      'order': 4, 'values': ['_Column_'],
                      'preformatted': False}
        
        dictEvent = {'parameter' : '_Event_', 'type' : 'char',
                     'label' : 'Event', 'length' : 8,
                     'order' : 5, 'values' : [ '_Event_' ],
                     'preformatted' : False}
        
        dictCutoff = {'parameter' : '_Cutoff_', 'type' : 'num',
                      'label' : 'Cutoff', 'length' : 8,
                      'order' : 6, 'values' : [ '_Cutoff_' ],
                      'preformatted' : False}
        
        dictSensitivity = {'parameter' : '_Sensitivity_', 'type' : 'num',
                           'label' : 'Sensitivity', 'length' : 8,
                           'order' : 7, 'values' : [ '_Sensitivity_' ],
                           'preformatted' : False}
        
        dictSpecificity = {'parameter' : '_Specificity_', 'type' : 'num',
                           'label' : 'Specificity', 'length' : 8,
                           'order' : 8, 'values' : [ '_Specificity_' ],
                           'preformatted' : False}
        
        dictFPR = {'parameter' : '_FPR_', 'type' : 'num',
                   'label' : 'False Positive Rate', 'length' : 8,
                   'order' : 9, 'values' : [ '_FPR_' ],
                   'preformatted' : False}
        
        dictOneMinusSpecificity = {'parameter' : '_OneMinusSpecificity_',
                                   'type' : 'num', 'label' : '1 - Specificity',
                                   'length' : 8, 'order' : 10,
                                   'values' : [ '_OneMinusSpecificity_' ],
                                   'preformatted' : False}
        
        parameterMap = {'_DataRole_': dictDataRole, '_PartInd_': dictPartInd,
                        '_PartInd__f':  dictPartIndf, '_Column_': dictColumn,
                        '_Event_': dictEvent, '_Cutoff_': dictCutoff,
                        '_Sensitivity_': dictSensitivity,
                        '_Specificity_': dictSpecificity, '_FPR_': dictFPR,
                        '_OneMinusSpecificity_': dictOneMinusSpecificity}
        
        dataPartitionExists = []   
        for i in range(3):
            if data[i][0] is not None:
                dataPartitionExists.append(i)
        
        listRoc = []
        numRows = 0
        
        for j in list(reversed(dataPartitionExists)):
            
            falsePosRate, truePosRate, threshold = metrics.roc_curve(data[j][0], data[j][1])
            rocDf = pd.DataFrame({'fpr': falsePosRate,
                                  'tpr': truePosRate,
                                  'threshold': np.minimum(1., np.maximum(0., threshold))})
            
            for count, row in rocDf.iterrows():
                
                rowStats = {}
                innerDict = {}
                
                if j==0:
                    dataRole = 'VALIDATE'
                    innerDict['_DataRole_'] = dataRole
                elif j==1:
                    dataRole = 'TRAIN'
                    innerDict['_DataRole_'] = dataRole
                elif j==2:
                    dataRole = 'TEST'
                    innerDict['_DataRole_'] = dataRole
                    
                innerDict.update({'_PartInd_': str(j),
                                '_PartInd__f': f'           {j}'})
        
                fpr = row['fpr']
                tpr = row['tpr']
                threshold = row['threshold']
        
                innerDict['_Column_'] = 'P_' + str(targetName) + '1'
                innerDict['_Event_'] = 1
                innerDict['_Cutoff_'] = threshold
                innerDict['_Sensitivity_'] = tpr
                innerDict['_Specificity_'] = (1.0 - fpr)
                innerDict['_FPR_'] = fpr
                innerDict['_OneMinusSpecificity_'] = fpr
        
                numRows += 1
                rowStats.update({'dataMap': innerDict,
                                 'rowNumber': numRows,
                                 'header': None})
                listRoc.append(dict(rowStats))
        
        outJSON = {'creationTimeStamp': None,
                   'modifiedTimeStamp': None,
                   'createdBy': None,
                   'modifiedBy': None,
                   'id': None,
                   'name': 'dmcas_roc',
                   'description': None,
                   'revision': 0,
                   'order': 0,
                   'type': None,
                   'parameterMap': parameterMap,
                   'data': listRoc,
                   'version': 1,
                   'xInteger': False,
                   'yInteger': False}
        
        with open(Path(jPath) / 'dmcas_roc.json', 'w') as jFile:
            json.dump(outJSON, jFile, indent=4)
            
    def generateLiftStat(self, data, targetName,
                           targetValue, jPath=Path.cwd()):
        '''
        Calculates the lift curves from user data and writes to a JSON file for
        importing to common model repository. Input data takes the form of a list of
        arrays, with the actual and predicted data separated into the
        following parititions: validate, train, test. An example is shown
        below:
            
            data = [(yValidateActual, yValidatePrediction),
                    (yTrainActual, yTrainPrediction),
                    (yTestActual, yTestPrediction)]
        
        For partitions without data, replace the array with a None assignment.
        
        Parameters
        ---------------
        data : list of arrays
            List of data arrays, separated into actual and predicted values
            and partitioned into Validate, Test, and Train sets. See above for 
            a formatted example.
        targetName: str
            Target variable name to be predicted.
        targetValue: int or float
            Value of target variable that indicates an event.
        jPath : string, optional
            Location for the output JSON file. The default is the current
            working directory.
        
        Yields
        ---------------
        'dmcas_lift.json'
            Output JSON file located at jPath.
        '''
        
        dictDataRole = {'parameter': '_DataRole_', 'type': 'char',
                        'label': 'Data Role', 'length': 10,
                        'order': 1, 'values': ['_DataRole_'],
                        'preformatted': False}

        dictPartInd = {'parameter': '_PartInd_', 'type': 'num',
                       'label': 'Partition Indicator', 'length': 8,
                       'order': 2, 'values': ['_PartInd_'],
                       'preformatted': False}
        
        dictPartIndf = {'parameter': '_PartInd__f', 'type': 'char',
                        'label': 'Formatted Partition', 'length': 12,
                        'order': 3, 'values': ['_PartInd__f'],
                        'preformatted': False}
        
        dictColumn = {'parameter' : '_Column_', 'type' : 'char',
                      'label' : 'Analysis Variable', 'length' : 32,
                      'order' : 4, 'values' : [ '_Column_' ],
                      'preformatted' : False}
        
        dictEvent = {'parameter' : '_Event_', 'type' : 'char',
                     'label' : 'Event', 'length' : 8,
                     'order' : 5, 'values' : [ '_Event_' ],
                     'preformatted' : False}
        
        dictDepth = {'parameter' : '_Depth_', 'type' : 'num',
                     'label' : 'Depth', 'length' : 8,
                     'order' : 7, 'values' : [ '_Depth_' ],
                     'preformatted' : False}
        
        dictNObs = {'parameter' : '_NObs_', 'type' : 'num',
                    'label' : 'Sum of Frequencies', 'length' : 8,
                    'order' : 8, 'values' : [ '_NObs_' ],
                    'preformatted' : False}
        
        dictGain = {'parameter' : '_Gain_', 'type' : 'num',
                    'label' : 'Gain', 'length' : 8,
                    'order' : 9, 'values' : [ '_Gain_' ],
                    'preformatted' : False}
        
        dictResp = {'parameter' : '_Resp_', 'type' : 'num',
                    'label' : '% Captured Response', 'length' : 8,
                    'order' : 10, 'values' : [ '_Resp_' ],
                    'preformatted' : False}
        
        dictCumResp = {'parameter' : '_CumResp_', 'type' : 'num',
                       'label' : 'Cumulative % Captured Response',
                       'length' : 8, 'order' : 11,
                       'values' : [ '_CumResp_' ], 'preformatted' : False}
        
        dictPctResp = {'parameter' : '_PctResp_', 'type' : 'num',
                       'label' : '% Response', 'length' : 8,
                       'order' : 12, 'values' : [ '_PctResp_' ],
                       'preformatted' : False}
        
        dictCumPctResp = {'parameter' : '_CumPctResp_', 'type' : 'num',
                          'label' : 'Cumulative % Response', 'length' : 8,
                          'order' : 13, 'values' : [ '_CumPctResp_' ],
                          'preformatted' : False}
        
        dictLift = {'parameter' : '_Lift_', 'type' : 'num',
                    'label' : 'Lift', 'length' : 8,
                    'order' : 14, 'values' : [ '_Lift_' ],
                    'preformatted' : False}
        
        dictCumLift = {'parameter' : '_CumLift_', 'type' : 'num',
                       'label' : 'Cumulative Lift', 'length' : 8,
                       'order' : 15, 'values' : [ '_CumLift_' ],
                       'preformatted' : False}
        
        parameterMap = {'_DataRole_': dictDataRole, '_PartInd_': dictPartInd,
                        '_PartInd__f': dictPartIndf, '_Column_': dictColumn,
                        '_Event_': dictEvent, '_Depth_': dictDepth,
                        '_NObs_': dictNObs, '_Gain_': dictGain,
                        '_Resp_': dictResp, '_CumResp_': dictCumResp,
                        '_PctResp_': dictPctResp,
                        '_CumPctResp_': dictCumPctResp,
                        '_Lift_': dictLift, '_CumLift_': dictCumLift}
        
        dataPartitionExists = []   
        for i in range(3):
            if data[i][0] is not None:
                dataPartitionExists.append(i)
        
        listLift = []
        numRows = 0
            
        for j in list(reversed(dataPartitionExists)):
            
            liftDf = self.calculateLift(pd.Series(data[j][0]), 1, data[j][1])
            
            for count, row in liftDf.iterrows():
                
                rowStats = {}
                innerDict = {}
                
                if j==0:
                    dataRole = 'VALIDATE'
                    innerDict['_DataRole_'] = dataRole
                elif j==1:
                    dataRole = 'TRAIN'
                    innerDict['_DataRole_'] = dataRole
                elif j==2:
                    dataRole = 'TEST'
                    innerDict['_DataRole_'] = dataRole
                    
                innerDict.update({'_PartInd_': str(j),
                                '_PartInd__f': f'           {j}'})
                
                quantileNumber = row['Quantile Number']
                gainNumber = row['Gain Number']
                gainPercent = row['Gain Percent']
                responsePercent = row['Response Percent']
                lift = row['Lift']
                
                accQuantilePercent = row['Acc Quantile Percent']
                accGainPercent = row['Acc Gain Percent']
                accResponsePercent = row['Acc Response Percent']
                accLift = row['Acc Lift']
                
                innerDict['_Column_'] = 'P_' + str(targetName) + '1'
                innerDict['_Event_'] = 1
                innerDict['_Depth_'] = accQuantilePercent
                innerDict['_NObs_'] = quantileNumber
                innerDict['_Gain_'] = gainNumber
                innerDict['_Resp_'] = gainPercent
                innerDict['_CumResp_'] = accGainPercent
                innerDict['_PctResp_'] = responsePercent
                innerDict['_CumPctResp_'] = accResponsePercent
                innerDict['_Lift_'] = lift
                innerDict['_CumLift_'] = accLift
                
                numRows += 1
                rowStats.update({'dataMap': innerDict,
                                 'rowNumber': numRows,
                                 'header': None})
                listLift.append(dict(rowStats))
            
        outJSON = {'creationTimeStamp': None,
                   'modifiedTimeStamp': None,
                   'createdBy': None,
                   'modifiedBy': None,
                   'id': None,
                   'name': 'dmcas_lift',
                   'description': None,
                   'revision': 0,
                   'order': 0,
                   'type': None,
                   'parameterMap': parameterMap,
                   'data': listLift,
                   'version': 1,
                   'xInteger': False,
                   'yInteger': False}
        
        with open(Path(jPath) / 'dmcas_lift.json', 'w') as jFile:
            json.dump(outJSON, jFile, indent=4)
        
    def calculateLift(self, actualValue, targetValue, predictValue):
        '''
        Calculates the lift statistics required to generate lift curves. 

        Parameters
        ----------
        actualValue : series
            Series containing the actual values of the target variable.
        targetValue: int or float
            Value of target variable that indicates an event.
        predictValue : array-like list
            Array-like list containing the predictions of the model for the 
            target variable

        Returns
        -------
        liftDf : dataframe
            Dataframe containing the lift and accumulated lift statistics.

        '''
        
        numObservations = len(actualValue)

        quantileCutOff = np.percentile(predictValue, np.arange(0, 100, 10))
        numQuantiles = len(quantileCutOff)
    
        quantileIndex = np.zeros(numObservations)
        for i in range(numObservations):
            k = numQuantiles
            for j in range(1, numQuantiles):
                if (predictValue[i] > quantileCutOff[-j]):
                    k -= 1
            quantileIndex[i] = k
    
        countTable = pd.crosstab(quantileIndex, actualValue)
        quantileNumber = countTable.sum(1)
        quantilePercent = 100 * (quantileNumber / numObservations)
        gainNumber = countTable[targetValue]
        totalNumResponse = gainNumber.sum(0)
        gainPercent = 100 * (gainNumber /totalNumResponse)
        responsePercent = 100 * (gainNumber / quantileNumber)
        overallResponsePercent = 100 * (totalNumResponse / numObservations)
        lift = responsePercent / overallResponsePercent
    
        liftDf = pd.DataFrame({'Quantile Number': quantileNumber,
                               'Quantile Percent': quantilePercent,
                               'Gain Number': gainNumber,
                               'Gain Percent': gainPercent,
                               'Response Percent': responsePercent,
                               'Lift': lift})
    
        accCountTable = countTable.cumsum(axis = 0)
        quantileNumber = accCountTable.sum(1)
        quantilePercent = 100 * (quantileNumber / numObservations)
        gainNumber = accCountTable[targetValue]
        gainPercent = 100 * (gainNumber / totalNumResponse)
        responsePercent = 100 * (gainNumber / quantileNumber)
        accLift = responsePercent / overallResponsePercent
    
    
        accLiftDf = pd.DataFrame({'Acc Quantile Number': quantileNumber,
                                  'Acc Quantile Percent': quantilePercent,
                                  'Acc Gain Number': gainNumber,
                                  'Acc Gain Percent': gainPercent,
                                  'Acc Response Percent': responsePercent,
                                  'Acc Lift': accLift})
    
        liftDf = pd.concat([liftDf, accLiftDf], axis=1, ignore_index=False)
        
        return(liftDf)
            
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
            
    def setInputType(self, paramValue):
        '''
        Given user input, determines if it is a string, int, or float. Returns
        the value cast to the determined type.
        
        Parameters
        ---------------
        paramValue : int, float, or str
            Value of the parameter.
            
        Returns
        ---------------
        paramValue : int, float, or str
            Value of the parameter.
        '''
        
        if paramValue.isdigit():
            return int(paramValue)
        try:
            float(paramValue)
            return float(paramValue)
        except:
            return paramValue
            
    def initializeDataMap(self, dataRole):
        '''
        Initializes a data map to include the data role, partition indicator,
        and formatted partition indicator.
        
        Parameters
        ---------------
        dataRole : int
            Identifier of the dataset's role. Either 1, 2, or 3.
            
        Returns
        ---------------
        dataMap : series
            Pandas series containing values and their parameter mappings.
        '''
        
        dataMap = pd.Series([self.convertDataRole(dataRole),
                             dataRole,
                             f'           {dataRole}'],
                            index=['_DataRole_',
                                   '_PartInd_',
                                   '_PartInd__f'])
        
        return dataMap
            
    def isNewParameter(self, paramName, parameterMap):
        '''
        Determines if parameter is already built in the parameter map, but
        first formats the parameter to the JSON standard (_<>_).
        
        Parameters
        ---------------
        paramName : string
            Name of the parameter.
        parameterMap : dict
            Dictionary of statistical parameters.
            
        Returns
        ---------------
        boolean
            True if the parameter is new. False if it already exists.
        '''
        
        paramName = self.formatParameter(paramName)
        
        if paramName in parameterMap.keys():
            return False
        else:
            return True
        
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
            
    def createParameterDict(self, paramName, paramLabel,
                            paramOrder, paramValue):
        '''
        Creates a parameter dictionary based on the provided values and return
        the dictionary.
        
        Parameters
        ---------------
        paramName : string
            Name of the parameter.
        paramLabel : string
            Description of the parameter.
        paramOrder : int
            Order to be displayed in SAS Open Model Manager.
        paramValue : int, float, or str
            Value of the parameter.
                
        Returns
        ---------------
        parameterDict : dict
            Output dictionary containing proper JSON formatted information.
        '''
        if isinstance(paramValue, (int, float)):
            paramType = 'num'
            paramLength = 8
        else:
            paramType = 'char'
            paramLength = len(paramValue)
        
        paramName = self.formatParameter(paramName)
        
        parameterDict = {'parameter': paramName,
                         'type': paramType,
                         'label': paramLabel,
                         'length': paramLength,
                         'order': paramOrder,
                         'values': [paramName],
                         'preformatted': False}
        
        return parameterDict
