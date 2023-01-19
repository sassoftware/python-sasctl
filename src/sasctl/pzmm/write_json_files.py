# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Standard Library Imports
import ast
import getpass
import importlib
import json
import math
import pickle
import pickletools
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path

# Third Party Imports
import pandas as pd


def flatten(nestedList):
    """
    Flatten a nested list.

    Flattens a nested list, while controlling for str values in list, such that the
    str values are not expanded into a list of single characters.

    Parameters
    ----------
    nestedList : list
        A nested list of strings.

    Yields
    ------
    list
        A flattened list of strings.
    """
    for item in nestedList:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


class JSONFiles:
    @staticmethod
    def writeVarJSON(inputData, isInput=True, jPath=Path.cwd()):
        """
        Writes a variable descriptor JSON file for input or output variables,
        based on input data containing predictor and prediction columns.

        This function creates a JSON file named either InputVar.json or
        OutputVar.json based on argument inputs.

        Parameters
        ----------
        inputData : dataframe or list of dicts
            Input dataframe containing the training data set in a pandas.Dataframe
            format. Columns are used to define predictor and prediction variables
            (ambiguously named "predict"). Providing a list of dict objects signals
            that the model files are being created from an MLFlow model.
        isInput : bool
            Boolean flag to check if generating the input or output variable JSON.
        jPath : string, optional
            File location for the output JSON file. Default is the current working
            directory.
        """
        outputJSON = pd.DataFrame()
        if isinstance(inputData, list):
            try:
                predictNames = [var["name"] for var in inputData]
            except KeyError:
                predictNames = [var["type"] for var in inputData]
            for i, name in enumerate(predictNames):
                if inputData[i]["type"] == "string":
                    isStr = True
                elif inputData[i]["type"] in ["double", "integer", "float", "long"]:
                    isStr = False
                elif inputData[i]["type"] == "tensor":
                    if inputData[i]["tensor-spec"]["dtype"] in "string":
                        isStr = True
                    else:
                        isStr = False

                if isStr:
                    outputLevel = "nominal"
                    outputType = "string"
                    outputLength = 8
                else:
                    outputLevel = "interval"
                    outputType = "decimal"
                    outputLength = 8
                outputRow = pd.Series(
                    [name, outputLength, outputType, outputLevel],
                    index=["name", "length", "type", "level"],
                )
                outputJSON = outputJSON.append([outputRow], ignore_index=True)
        else:
            try:
                predictNames = inputData.columns.values.tolist()
                isSeries = False
            except AttributeError:
                predictNames = [inputData.name]
                isSeries = True

            # loop through all predict variables to determine their name, length,
            # type, and level; append each to outputJSON
            for name in predictNames:
                if isSeries:
                    predict = inputData
                else:
                    predict = inputData[name]
                firstRow = predict.loc[predict.first_valid_index()]
                dType = predict.dtypes.name
                isStr = type(firstRow) is str

                if isStr:
                    outputLevel = "nominal"
                    outputType = "string"
                    outputLength = predict.str.len().max()
                else:
                    if dType == "category":
                        outputLevel = "nominal"
                    else:
                        outputLevel = "interval"
                    outputType = "decimal"
                    outputLength = 8

                outputRow = pd.Series(
                    [name, outputLength, outputType, outputLevel],
                    index=["name", "length", "type", "level"],
                )
                outputJSON = outputJSON.append([outputRow], ignore_index=True)

        if isInput:
            fileName = "inputVar.json"
        else:
            fileName = "outputVar.json"

        with open(Path(jPath) / fileName, "w") as jFile:
            dfDump = pd.DataFrame.to_dict(outputJSON.transpose()).values()
            json.dump(list(dfDump), jFile, indent=4, skipkeys=True)
        print(
            "{} was successfully written and saved to {}".format(
                fileName, Path(jPath) / fileName
            )
        )

    @staticmethod
    def writeModelPropertiesJSON(
        modelName,
        modelDesc,
        targetVariable,
        modelType,
        modelPredictors,
        targetEvent,
        numTargetCategories,
        eventProbVar=None,
        jPath=Path.cwd(),
        modeler=None,
    ):
        """
        Writes a JSON file containing SAS Model Manager model properties.

        The JSON file format is required by the SAS Model Repository API service and
        only eventProbVar can be 'None'. This function outputs a JSON file located
        named "ModelProperties.json".

        Parameters
        ----------
        modelName : string
            User-defined model name.
        modelDesc : string
            User-defined model description.
        targetVariable : string
            Target variable to be predicted by the model.
        modelType : string
            User-defined model type.
        modelPredictors : list of strings
            List of predictor variables for the model.
        targetEvent : string
            Model target event (for example, 1 for a binary event).
        numTargetCategories : int
            Number of possible target categories (for example, 2 for a binary event).
        eventProbVar : string, optional
            Model prediction metric for scoring. Default is None.
        jPath : string, optional
            Location for the output JSON file. The default is the current working
            directory.
        modeler : string, optional
            The modeler name to be displayed in the model properties. The default value
            is None.
        """
        # Check if model description provided is smaller than the 1024-character limit
        if len(modelDesc) > 1024:
            modelDesc = modelDesc[:1024]
            warnings.warn(
                "WARNING: The provided model description was truncated to 1024 "
                "characters. "
            )

        if numTargetCategories > 2 and not targetEvent:
            targetLevel = "NOMINAL"
        elif numTargetCategories > 2 and targetEvent:
            targetLevel = "ORDINAL"
        else:
            targetLevel = "BINARY"
            targetEvent = 1

        if eventProbVar is None:
            try:
                eventProbVar = "P_" + targetVariable + targetEvent
            except TypeError:
                eventProbVar = None
        # Replace <myUserID> with the user ID of the modeler that created the model.
        if modeler is None:
            try:
                modeler = getpass.getuser()
            except OSError:
                modeler = "<myUserID>"

        pythonVersion = sys.version.split(" ", 1)[0]

        propIndex = [
            "name",
            "description",
            "function",
            "scoreCodeType",
            "trainTable",
            "trainCodeType",
            "algorithm",
            "targetVariable",
            "targetEvent",
            "targetLevel",
            "eventProbVar",
            "modeler",
            "tool",
            "toolVersion",
        ]

        modelProperties = [
            modelName,
            modelDesc,
            "classification",
            "python",
            " ",
            "Python",
            modelType,
            targetVariable,
            targetEvent,
            targetLevel,
            eventProbVar,
            modeler,
            "Python 3",
            pythonVersion,
        ]

        outputJSON = pd.Series(modelProperties, index=propIndex)

        with open(Path(jPath) / "ModelProperties.json", "w") as jFile:
            dfDump = pd.Series.to_dict(outputJSON.transpose())
            json.dump(dfDump, jFile, indent=4, skipkeys=True)
        print(
            "{} was successfully written and saved to {}".format(
                "ModelProperties.json", Path(jPath) / "ModelProperties.json"
            )
        )

    @staticmethod
    def writeFileMetadataJSON(modelPrefix, jPath=Path.cwd(), isH2OModel=False):
        """
        Writes a file metadata JSON file pointing to all relevant files.

        This function outputs a JSON file named "fileMetadata.json".

        Parameters
        ----------
        modelPrefix : string
            The variable for the model name that is used when naming model files. (For
            example, hmeqClassTree + [Score.py | .pickle]).
        jPath : string, optional
            Location for the output JSON file. The default value is the current working
            directory.
        isH2OModel : boolean, optional
            Sets whether the model metadata is associated with an H2O.ai model. If set
            as True, the MOJO model file will be set as a score resource. The default
            value is False.
        """
        if not isH2OModel:
            fileMetadata = pd.DataFrame(
                [
                    ["inputVariables", "inputVar.json"],
                    ["outputVariables", "outputVar.json"],
                    ["score", modelPrefix + "Score.py"],
                    ["scoreResource", modelPrefix + ".pickle"],
                ],
                columns=["role", "name"],
            )
        else:
            fileMetadata = pd.DataFrame(
                [
                    ["inputVariables", "inputVar.json"],
                    ["outputVariables", "outputVar.json"],
                    ["score", modelPrefix + "Score.py"],
                    ["scoreResource", modelPrefix + ".mojo"],
                ],
                columns=["role", "name"],
            )

        with open(Path(jPath) / "fileMetadata.json", "w") as jFile:
            dfDump = pd.DataFrame.to_dict(fileMetadata.transpose()).values()
            json.dump(list(dfDump), jFile, indent=4, skipkeys=True)
        print(
            "{} was successfully written and saved to {}".format(
                "fileMetaData.json", Path(jPath) / "fileMetaData.json"
            )
        )

    @classmethod
    def writeBaseFitStat(
        cls, csvPath=None, jPath=Path.cwd(), userInput=False, tupleList=None
    ):
        """
        Writes a JSON file to display fit statistics for the model in SAS Model Manager.

        There are three modes to add fit parameters to the JSON file:

            1. Call the function with additional tuple arguments containing
            the name of the parameter, its value, and the partition that it
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

        This function outputs a JSON file named "dmcas_fitstat.json".

        Parameters
        ----------
        csvPath : string, optional
            Location for an input CSV file that contains parameter statistics. The
            default value is None.
        jPath : string, optional
            Location for the output JSON file. The default value is the current working
            directory.
        userInput : boolean, optional
            If true, prompt the user for more parameters. The default value is false.
        tupleList : list of tuples, optional
            Input parameter tuples in the form of (parameterName, parameterLabel,
            parameterValue, dataRole). For example, a sample parameter call would be
            'NObs', 'Sum of Frequencies', 3488, or 'TRAIN'. Variable dataRole
            is typically either TRAIN, TEST, or VALIDATE or 1, 2, 3 respectively. The
            default value is None.
        """
        validParams = [
            "_RASE_",
            "_NObs_",
            "_GINI_",
            "_GAMMA_",
            "_MCE_",
            "_ASE_",
            "_MCLL_",
            "_KS_",
            "_KSPostCutoff_",
            "_DIV_",
            "_TAU_",
            "_KSCut_",
            "_C_",
        ]

        nullJSONPath = Path(__file__).resolve().parent / "null_dmcas_fitstat.json"
        nullJSONDict = cls.readJSONFile(nullJSONPath)

        dataMap = [{}, {}, {}]
        for i in range(3):
            dataMap[i] = nullJSONDict["data"][i]

        if tupleList is not None:
            for paramTuple in tupleList:
                # ignore incorrectly formatted input arguments
                if type(paramTuple) == tuple and len(paramTuple) == 3:
                    paramName = cls.formatParameter(paramTuple[0])
                    if paramName not in validParams:
                        continue
                    if type(paramTuple[2]) == str:
                        dataRole = cls.convertDataRole(paramTuple[2])
                    else:
                        dataRole = paramTuple[2]
                    dataMap[dataRole - 1]["dataMap"][paramName] = paramTuple[1]

        if userInput:
            while True:
                paramName = input("Parameter name: ")
                paramName = cls.formatParameter(paramName)
                if paramName not in validParams:
                    print("Not a valid parameter. Please see documentation.")
                    if input("More parameters? (Y/N)") == "N":
                        break
                    continue
                paramValue = input("Parameter value: ")
                dataRole = input("Data role: ")

                if type(dataRole) is str:
                    dataRole = cls.convertDataRole(dataRole)
                dataMap[dataRole - 1]["dataMap"][paramName] = paramValue

                if input("More parameters? (Y/N)") == "N":
                    break

        if csvPath is not None:
            csvData = pd.read_csv(csvPath)
            for i, row in enumerate(csvData.values):
                paramName, paramValue, dataRole = row
                paramName = cls.formatParameter(paramName)
                if paramName not in validParams:
                    continue
                if type(dataRole) is str:
                    dataRole = cls.convertDataRole(dataRole)
                dataMap[dataRole - 1]["dataMap"][paramName] = paramValue

        outJSON = nullJSONDict
        for i in range(3):
            outJSON["data"][i] = dataMap[i]

        with open(Path(jPath) / "dmcas_fitstat.json", "w") as jFile:
            json.dump(outJSON, jFile, indent=4, skipkeys=True)
        print(
            "{} was successfully written and saved to {}".format(
                "dmcas_fitstat.json", Path(jPath) / "dmcas_fitstat.json"
            )
        )

    @classmethod
    def calculateFitStat(
        cls, validateData=None, trainData=None, testData=None, jPath=Path.cwd()
    ):
        """
        Calculates fit statistics from user data and predictions and then writes to a
        JSON file for importing into the common model repository.

        Note that if no data set is provided (validate, train, or test),
        this function raises an error and does not create a JSON file.

        Datasets can be provided in the following forms:
        * pandas dataframe; the actual and predicted values are their own columns
        * numpy array; the actual and predicted values are their own columns or rows and
        ordered such that the actual values come first and the predicted second
        * list; the actual and predicted values are their own indexed entry

        This function outputs a JSON file named "dmcas_fitstat.json".

        Parameters
        ----------
        validateData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the validation data set, including both
            the actual and predicted values. The default value is None.
        trainData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the train data set, including both
            the actual and predicted values. The default value is None.
        testData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the test data set, including both
            the actual and predicted values. The default value is None.
        jPath : string, optional
            Location for the output JSON file. The default value is the current
            working directory.
        """
        # If numpy inputs are supplied, then assume numpy is installed
        try:
            import numpy as np
        except ImportError:
            np = None

        try:
            from sklearn import metrics
        except ImportError:
            raise RuntimeError(
                "The 'scikit-learn' package is required to use the calculateFitStat "
                "function. "
            )

        nullJSONPath = Path(__file__).resolve().parent / "null_dmcas_fitstat.json"
        nullJSONDict = cls.readJSONFile(nullJSONPath)

        dataSets = [[[None], [None]], [[None], [None]], [[None], [None]]]

        dataPartitionExists = []
        for i, data in enumerate([validateData, trainData, testData]):
            if data is not None:
                dataPartitionExists.append(i)
                if type(data) is pd.core.frame.DataFrame:
                    dataSets[i] = data.transpose().values.tolist()
                elif type(data) is list:
                    dataSets[i] = data
                elif type(data) is np.ndarray:
                    dataSets[i] = data.tolist()

        if len(dataPartitionExists) == 0:
            raise ValueError(
                "No data was provided. Please provide the actual and predicted values "
                "for at least one of the partitions (VALIDATE, TRAIN, or TEST)."
            )

        for j in dataPartitionExists:
            fitStats = nullJSONDict["data"][j]["dataMap"]

            fitStats["_PartInd_"] = j

            # If the data provided is Predicted | Actual instead of Actual |
            # Predicted, catch the error and flip the columns
            try:
                fpr, tpr, _ = metrics.roc_curve(dataSets[j][0], dataSets[j][1])
            except ValueError:
                tempSet = dataSets[j]
                dataSets[j][0] = tempSet[1]
                dataSets[j][1] = tempSet[0]
                fpr, tpr, _ = metrics.roc_curve(dataSets[j][0], dataSets[j][1])

            RASE = math.sqrt(metrics.mean_squared_error(dataSets[j][0], dataSets[j][1]))
            fitStats["_RASE_"] = RASE

            NObs = len(dataSets[j][0])
            fitStats["_NObs_"] = NObs

            auc = metrics.roc_auc_score(dataSets[j][0], dataSets[j][1])
            GINI = (2 * auc) - 1
            fitStats["_GINI_"] = GINI

            try:
                from scipy.stats import gamma

                _, _, scale = gamma.fit(dataSets[j][1])
                fitStats["_GAMMA_"] = 1 / scale
            except ImportError:
                warnings.warn(
                    "scipy was not installed, so the gamma calculation could"
                    "not be computed."
                )
                fitStats["_GAMMA_"] = None

            intPredict = [round(x) for x in dataSets[j][1]]
            MCE = 1 - metrics.accuracy_score(dataSets[j][0], intPredict)
            fitStats["_MCE_"] = MCE

            ASE = metrics.mean_squared_error(dataSets[j][0], dataSets[j][1])
            fitStats["_ASE_"] = ASE

            MCLL = metrics.log_loss(dataSets[j][0], dataSets[j][1])
            fitStats["_MCLL_"] = MCLL

            KS = max(abs(fpr - tpr))
            fitStats["_KS_"] = KS

            KSPostCutoff = None
            fitStats["_KSPostCutoff_"] = KSPostCutoff

            DIV = len(dataSets[j][0])
            fitStats["_DIV_"] = DIV

            TAU = pd.Series(dataSets[j][0]).corr(
                pd.Series(dataSets[j][1]), method="kendall"
            )
            fitStats["_TAU_"] = TAU

            KSCut = None
            fitStats["_KSCut_"] = KSCut

            C = metrics.auc(fpr, tpr)
            fitStats["_C_"] = C

            nullJSONDict["data"][j]["dataMap"] = fitStats

        with open(Path(jPath) / "dmcas_fitstat.json", "w") as jFile:
            json.dump(nullJSONDict, jFile, indent=4)
        print(
            "{} was successfully written and saved to {}".format(
                "dmcas_fitstat.json", Path(jPath) / "dmcas_fitstat.json"
            )
        )

    @classmethod
    def generateROCLiftStat(
        cls,
        targetName,
        targetValue,
        swatConn,
        validateData=None,
        trainData=None,
        testData=None,
        jPath=Path.cwd(),
    ):
        """
        Calculates the ROC and Lift curves from user data and model predictions and
        the writes it to JSON files for importing in to the common model repository.

        ROC and Lift calculations are completed by CAS through a SWAT call. Note that
        if no data set is provided (validate, train, or test), this function raises
        an error and does not create any JSON files.

        This function outputs a pair of JSON files named "dmcas_lift.json" and
        "dmcas_roc.json".

        Parameters
        ---------------
        targetName: str
            Target variable name to be predicted.
        targetValue: int or float
            Value of target variable that indicates an event.
        swatConn: SWAT connection to CAS
            Connection object to CAS service in SAS Model Manager through SWAT
            authentication.
        validateData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the validation data set, including both the
            actual values and the calculated probabilities. The default value is None.
        trainData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the train data set, including both the actual
            values and the calculated probabilities. The default value is None.
        testData : pandas dataframe, numpy array, or list, optional
            Dataframe, array, or list of the test data set, including both the actual
            values and the calculated probabilities. The default value is None.
        jPath : string, optional
            Location for the output JSON file. The default value is the current working
            directory.
        """
        # If numpy inputs are supplied, then assume numpy is installed
        try:
            # noinspection PyPackageRequirements
            import numpy as np
        except ImportError:
            np = None
        try:
            import swat
        except ImportError:
            raise RuntimeError(
                "The 'swat' package is required to generate ROC and Lift charts with "
                "this function. "
            )

        nullJSONROCPath = Path(__file__).resolve().parent / "null_dmcas_roc.json"
        nullJSONROCDict = cls.readJSONFile(nullJSONROCPath)

        nullJSONLiftPath = Path(__file__).resolve().parent / "null_dmcas_lift.json"
        nullJSONLiftDict = cls.readJSONFile(nullJSONLiftPath)

        dataSets = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
        columns = ["actual", "predict"]

        dataPartitionExists = []
        # Check if a data partition exists, then convert to a pandas dataframe
        for i, data in enumerate([validateData, trainData, testData]):
            if data is not None:
                dataPartitionExists.append(i)
                if type(data) is list:
                    dataSets[i][columns] = list(zip(*data))
                elif type(data) is pd.core.frame.DataFrame:
                    try:
                        dataSets[i][columns[0]] = data.iloc[:, 0]
                        dataSets[i][columns[1]] = data.iloc[:, 1]
                    except NameError:
                        dataSets[i] = pd.DataFrame(data=data.iloc[:, 0]).rename(
                            columns={data.columns[0]: columns[0]}
                        )
                        dataSets[i][columns[1]] = data.iloc[:, 1]
                elif type(data) is np.ndarray:
                    try:
                        dataSets[i][columns] = data
                    except ValueError:
                        dataSets[i][columns] = data.transpose()

        if len(dataPartitionExists) == 0:
            raise ValueError(
                "No data was provided. Please provide the actual and predicted values "
                "for at least one of the partitions (VALIDATE, TRAIN, or TEST)"
            )

        nullLiftRow = list(range(1, 64))
        nullROCRow = list(range(1, 301))

        swatConn.loadactionset("percentile")

        for i in dataPartitionExists:
            swatConn.read_frame(
                dataSets[i][columns], casout=dict(name="SCOREDVALUES", replace=True)
            )
            swatConn.percentile.assess(
                table="SCOREDVALUES",
                inputs=[columns[1]],
                casout=dict(name="SCOREASSESS", replace=True),
                response=columns[0],
                event=str(targetValue),
            )
            assessROC = swatConn.CASTable("SCOREASSESS_ROC").to_frame()
            assessLift = swatConn.CASTable("SCOREASSESS").to_frame()

            for j in range(100):
                rowNumber = (i * 100) + j
                nullROCRow.remove(rowNumber + 1)
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_Event_"] = targetValue
                nullJSONROCDict["data"][rowNumber]["dataMap"][
                    "_TargetName_"
                ] = targetName
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_Cutoff_"] = str(
                    assessROC["_Cutoff_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_TP_"] = str(
                    assessROC["_TP_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_FP_"] = str(
                    assessROC["_FP_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_FN_"] = str(
                    assessROC["_FN_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_TN_"] = str(
                    assessROC["_TN_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_Sensitivity_"] = str(
                    assessROC["_Sensitivity_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_Specificity_"] = str(
                    assessROC["_Specificity_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_KS_"] = str(
                    assessROC["_KS_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_KS2_"] = str(
                    assessROC["_KS2_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_FHALF_"] = str(
                    assessROC["_FHALF_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_FPR_"] = str(
                    assessROC["_FPR_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_ACC_"] = str(
                    assessROC["_ACC_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_FDR_"] = str(
                    assessROC["_FDR_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_F1_"] = str(
                    assessROC["_F1_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_C_"] = str(
                    assessROC["_C_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_GINI_"] = str(
                    assessROC["_GINI_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_GAMMA_"] = str(
                    assessROC["_GAMMA_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_TAU_"] = str(
                    assessROC["_TAU_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"]["_MiscEvent_"] = str(
                    assessROC["_MiscEvent_"][j]
                )
                nullJSONROCDict["data"][rowNumber]["dataMap"][
                    "_OneMinusSpecificity_"
                ] = str(1 - assessROC["_Specificity_"][j])

            for j in range(21):
                rowNumber = (i * 21) + j
                nullLiftRow.remove(rowNumber + 1)
                nullJSONLiftDict["data"][rowNumber]["dataMap"]["_Event_"] = str(
                    targetValue
                )
                nullJSONLiftDict["data"][rowNumber]["dataMap"][
                    "_TargetName_"
                ] = targetName
                if j != 0:
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_Depth_"] = str(
                        assessLift["_Depth_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_Value_"] = str(
                        assessLift["_Value_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_NObs_"] = str(
                        assessLift["_NObs_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_NEvents_"] = str(
                        assessLift["_NEvents_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"][
                        "_NEventsBest_"
                    ] = str(assessLift["_NEventsBest_"][j - 1])
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_Resp_"] = str(
                        assessLift["_Resp_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_RespBest_"] = str(
                        assessLift["_RespBest_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_Lift_"] = str(
                        assessLift["_Lift_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_LiftBest_"] = str(
                        assessLift["_LiftBest_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_CumResp_"] = str(
                        assessLift["_CumResp_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"][
                        "_CumRespBest_"
                    ] = str(assessLift["_CumRespBest_"][j - 1])
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_CumLift_"] = str(
                        assessLift["_CumLift_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"][
                        "_CumLiftBest_"
                    ] = str(assessLift["_CumLiftBest_"][j - 1])
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_PctResp_"] = str(
                        assessLift["_PctResp_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"][
                        "_PctRespBest_"
                    ] = str(assessLift["_PctRespBest_"][j - 1])
                    nullJSONLiftDict["data"][rowNumber]["dataMap"][
                        "_CumPctResp_"
                    ] = str(assessLift["_CumPctResp_"][j - 1])
                    nullJSONLiftDict["data"][rowNumber]["dataMap"][
                        "_CumPctRespBest_"
                    ] = str(assessLift["_CumPctRespBest_"][j - 1])
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_Gain_"] = str(
                        assessLift["_Gain_"][j - 1]
                    )
                    nullJSONLiftDict["data"][rowNumber]["dataMap"]["_GainBest_"] = str(
                        assessLift["_GainBest_"][j - 1]
                    )

        # If not all partitions are present, clean up the dicts for compliant formatting
        if len(dataPartitionExists) < 3:
            # Remove missing partitions from ROC and Lift dicts
            for index, row in reversed(list(enumerate(nullJSONLiftDict["data"]))):
                if int(row["rowNumber"]) in nullLiftRow:
                    nullJSONLiftDict["data"].pop(index)
            for index, row in reversed(list(enumerate(nullJSONROCDict["data"]))):
                if int(row["rowNumber"]) in nullROCRow:
                    nullJSONROCDict["data"].pop(index)

            # Reassign the row number values to match what is left in each dict
            for i, _ in enumerate(nullJSONLiftDict["data"]):
                nullJSONLiftDict["data"][i]["rowNumber"] = i + 1
            for i, _ in enumerate(nullJSONROCDict["data"]):
                nullJSONROCDict["data"][i]["rowNumber"] = i + 1

        with open(Path(jPath) / "dmcas_roc.json", "w") as jFile:
            json.dump(nullJSONROCDict, jFile, indent=4)
        print(
            "{} was successfully written and saved to {}".format(
                "dmcas_roc.json", Path(jPath) / "dmcas_roc.json"
            )
        )

        with open(Path(jPath) / "dmcas_lift.json", "w") as jFile:
            json.dump(nullJSONLiftDict, jFile, indent=4)
        print(
            "{} was successfully written and saved to {}".format(
                "dmcas_lift.json", Path(jPath) / "dmcas_lift.json"
            )
        )

    @staticmethod
    def readJSONFile(path):
        """
        Reads a JSON file from a given path.

        Parameters
        ----------
        path : str or pathlib Path
            Location of the JSON file to be opened.

        Returns
        -------
        json.load(jFile) : str
            String contents of JSON file.
        """
        with open(path) as jFile:
            return json.load(jFile)

    @staticmethod
    def formatParameter(paramName):
        """
        Formats the parameter name to the JSON standard.

        Note that no changes are applied if the string is already formatted correctly.

        Parameters
        ----------
        paramName : string
            Name of the parameter.

        Returns
        -------
        paramName : string
            Name of the parameter.
        """
        if not (paramName.startswith("_") and paramName.endswith("_")):
            if not paramName.startswith("_"):
                paramName = "_" + paramName
            if not paramName.endswith("_"):
                paramName = paramName + "_"

        return paramName

    @staticmethod
    def convertDataRole(dataRole):
        """
        Converts the data role identifier from string to int or int to string.

        JSON file descriptors require the string, int, and formatted int. If the
        provided data role is not valid, defaults to TRAIN (1).

        Parameters
        ----------
        dataRole : string or int
            Identifier of the data set's role; either TRAIN, TEST, or VALIDATE, which
            correspond to 1, 2, or 3.

        Returns
        -------
        conversion : int or string
            Converted data role identifier.
        """
        if type(dataRole) is int or type(dataRole) is float:
            dataRole = int(dataRole)
            if dataRole == 1:
                conversion = "TRAIN"
            elif dataRole == 2:
                conversion = "TEST"
            elif dataRole == 3:
                conversion = "VALIDATE"
            else:
                conversion = "TRAIN"
        elif type(dataRole) is str:
            if dataRole == "TRAIN":
                conversion = 1
            elif dataRole == "TEST":
                conversion = 2
            elif dataRole == "VALIDATE":
                conversion = 3
            else:
                conversion = 1
        else:
            conversion = 1

        return conversion

    @classmethod
    def create_requirements_json(cls, model_path=Path.cwd()):
        """
        Searches the model directory for Python scripts and pickle files and
        determines their Python package dependencies.

        Found dependencies are then matched to the package version found in the
        current working environment. Then the package and version are written to a
        requirements.json file.

        WARNING: The methods utilized in this function can determine package
        dependencies from provided scripts and pickle files, but CANNOT determine the
        required package versions without being in the development environment which
        they were originally created.

        This function works best when run in the model development environment and is
        likely to throw errors if run in another environment (and/or produce
        incorrect package versions). In the case of using this function outside the
        model development environment, it is recommended to the user that they adjust
        the requirements.json file's package versions to match the model development
        environment.

        This function outputs a JSON file named "requirements.json".

        Parameters
        ----------
        model_path : str, optional
            The path to a Python project, by default the current working directory.

        Returns ------- list of dicts List of dictionary representations of the json
        file contents, split into each package and/or warning.
        """
        pickle_packages = []
        pickle_files = cls.get_pickle_file(model_path)
        for pickle_file in pickle_files:
            pickle_packages.append(cls.get_pickle_dependencies(pickle_file))

        code_dependencies = cls.get_code_dependencies(model_path)

        package_list = list(pickle_packages) + code_dependencies
        package_list = list(set(list(flatten(package_list))))
        package_list = cls.remove_standard_library_packages(package_list)
        package_and_version = cls.get_local_package_version(package_list)
        # Identify packages with missing versions
        missing_package_versions = [
            item[0] for item in package_and_version if not item[1]
        ]

        # Create a list of dicts related to each package or warning
        json_dicts = []
        if missing_package_versions:
            json_dicts.append(
                {
                    "Warning": "The existence and/or versions for the following "
                    "packages could not be determined:",
                    "Packages": ", ".join(missing_package_versions),
                }
            )
        for package, version in package_and_version:
            if version:
                json_dicts.append(
                    {
                        "step": f"install {package}",
                        "command": f"pip install {package}=={version}",
                    }
                )
        with open(  # skipcq: PTC-W6004
            Path(model_path) / "requirements.json", "w"
        ) as file:
            file.write(json.dumps(json_dicts, indent=4))

        return json_dicts

    @staticmethod
    def get_local_package_version(package_list):
        """
        Get package_name versions from the local environment.

        If the package_name does not contain an attribute of "__version__",
        "version", or "VERSION", no package_name version will be found.

        Parameters
        ----------
        package_list : list
            List of Python packages.

        Returns
        -------
        list
            Nested list of Python package_name names and found versions.
        """

        def package_not_found_output(package_name, package_versions):
            warnings.warn(
                f"Warning: Package {package_name} was not found in the local "
                f"environment. Either {package_name} is not a valid Python package, "
                f"or the package is not present in this environment. The "
                f"requirements.json file  will include a commented out version of the "
                f"pip installation command at the bottom of the file. Please review "
                f"the file and verify that the package exists and input the version "
                f"needed. "
            )
            package_versions.append([package_name, None])
            return package_versions

        package_and_version = []

        for package in package_list:
            try:
                name = importlib.import_module(package)
                try:
                    package_and_version.append([package, name.__version__])
                except AttributeError:
                    try:
                        package_and_version.append([package, name.version])
                    except AttributeError:
                        try:
                            package_and_version.append([package, name.VERSION])
                        except AttributeError:
                            package_and_version = package_not_found_output(
                                package, package_and_version
                            )
            except ModuleNotFoundError:
                package_and_version = package_not_found_output(
                    package, package_and_version
                )

        return package_and_version

    @classmethod
    def get_code_dependencies(cls, model_path=Path.cwd()):
        """
        Get the package dependencies for all Python scripts in the provided directory
        path.

        Note that currently this functionality only works for .py files.

        Parameters
        ----------
        model_path : string, optional
            File location for the output JSON file. Default is the current working
            directory.

        Returns
        -------
        list
            List of found package dependencies.
        """
        import_info = []
        for file in sorted(Path(model_path).glob("*.py")):
            import_info.append(cls.find_imports(file))
        import_info = list(set(flatten(import_info)))
        return import_info

    @staticmethod
    def find_imports(file_path):
        """
        Find import calls in provided Python code path.

        Ignores built in Python modules.

        Credit: modified from https://stackoverflow.com/questions/44988487/regex-to
        -parse-import-statements-in-python

        Parameters
        ----------
        file_path : string or Path
            File location for the Python file to be parsed.

        Returns
        -------
        list
            List of found package dependencies.
        """
        with open(file_path, "r") as file:  # skipcq: PTC-W6004
            file_text = file.read()
            # Parse the file to get the abstract syntax tree representation
            tree = ast.parse(file_text)
            modules = []

            # Walk through each node in the ast to find import calls
            for node in ast.walk(tree):
                # Determine parent module for `from * import *` calls
                if isinstance(node, ast.ImportFrom):
                    modules.append(node.module)
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        modules.append(name.name)

        modules = list(set(modules))
        try:
            # Remove 'settings' module generated for SAS Model Manager score code
            modules.remove("settings")
            return modules
        except ValueError:
            return modules

    @staticmethod
    def get_pickle_file(pickle_folder=Path.cwd()):
        """
        Given a file path, retrieve the pickle file(s).

        Parameters
        ----------
        pickle_folder : str
            File location for the input pickle file. Default is the current working
            directory.

        Returns
        -------
        list
            A list of pickle files.
        """
        return [
            p for p in Path(pickle_folder).iterdir() if p.suffix in [".pickle", ".pkl"]
        ]

    @classmethod
    def get_pickle_dependencies(cls, pickle_file):
        """
        Reads the pickled byte stream from a file object, serializes the pickled byte
        stream as a bytes object, and inspects the bytes object for all Python
        modules and aggregates them in a list.

        Parameters
        ----------
        pickle_file : str
            The file where you stored pickle data.

        Returns
        -------
        list
            A list of modules obtained from the pickle stream. Duplicates are removed
            and Python built-in modules are removed.
        """
        with (open(pickle_file, "rb")) as open_file:  # skipcq: PTC-W6004
            obj = pickle.load(open_file)  # skipcq: BAN-B301
            dumps = pickle.dumps(obj)

        modules = cls.get_package_names(dumps)
        return modules

    @staticmethod
    def get_package_names(stream):
        """
        Generates a list of found `package` names from a pickle stream.

        In most cases, the `packages` returned by the function will be valid Python
        packages. A check is made in get_local_package_version to ensure that the
        package is in fact a valid Python package.

        This code has been adapted from the following stackoverflow example and
        utilizes the pickletools package.
        Credit: modified from
        https://stackoverflow.com/questions/64850179/inspecting-a-pickle-dump-for
        -dependencies
        More information here:
        https://github.com/python/cpython/blob/main/Lib/pickletools.py

        Parameters
        ----------
        stream : bytes or str
            A file like object or string containing the pickle.

        Returns
        -------
        list
            List of package names found as module dependencies in the pickle file.
        """
        # Collect opcodes, arguments, and position values from the pickle stream
        opcode, arg, pos = [], [], []
        for o, a, p in pickletools.genops(stream):
            opcode.append(o.name)
            arg.append(a)
            pos.append(p)

        # Convert to a pandas dataframe for ease of conditional filtering
        df_pickle = pd.DataFrame({"opcode": opcode, "arg": arg, "pos": pos})

        # For all opcodes labelled GLOBAL or STACK_GLOBAL pull out the package names
        global_stack = df_pickle[
            (df_pickle.opcode == "GLOBAL") | (df_pickle.opcode == "STACK_GLOBAL")
        ]
        # From the argument column, split the string of the form `X.Y.Z` by `.` and
        # return only the unique `X's`
        stack_packages = (
            global_stack.arg.str.split().str[0].str.split(".").str[0].unique().tolist()
        )

        # For all opcodes labelled BINUNICODE or SHORT_BINUNICODE grab the package names
        binunicode = df_pickle[
            (df_pickle.opcode == "BINUNICODE")
            | (df_pickle.opcode == "SHORT_BINUNICODE")
        ]
        # From the argument column, split the string by `.`, then return only unique
        # cells with at least one split
        arg_binunicode = binunicode.arg.str.split(".")
        unicode_packages = (
            arg_binunicode.loc[arg_binunicode.str.len() > 1].str[0].unique().tolist()
        )
        # Remove invalid `package` names from the list
        unicode_packages = [x for x in unicode_packages if x.isidentifier()]

        # Combine the two package lists and remove any duplicates
        packages = list(set(stack_packages + unicode_packages))

        # Return the package list without any None values
        return [x for x in packages if x]

    @staticmethod
    def remove_standard_library_packages(package_list):
        """
        Remove any packages from the required list of installed packages that are
        part of the Python Standard Library.

        Parameters
        ----------
        package_list : list
            List of all packages found that are not Python built-in packages.

        Returns
        -------
        list
            List of all packages found that are not Python built-in packages or part of
            the Python Standard Library.
        """
        py10stdlib = [
            "_aix_support",
            "_heapq",
            "lzma",
            "gc",
            "mailcap",
            "winsound",
            "sre_constants",
            "netrc",
            "audioop",
            "xdrlib",
            "code",
            "_pyio",
            "_gdbm",
            "unicodedata",
            "pwd",
            "xml",
            "_symtable",
            "pkgutil",
            "_decimal",
            "_compat_pickle",
            "_frozen_importlib_external",
            "_signal",
            "fcntl",
            "wsgiref",
            "uu",
            "textwrap",
            "_codecs_iso2022",
            "keyword",
            "distutils",
            "binascii",
            "email",
            "reprlib",
            "cmd",
            "cProfile",
            "dataclasses",
            "_sha512",
            "ntpath",
            "readline",
            "signal",
            "_elementtree",
            "dis",
            "rlcompleter",
            "_json",
            "_ssl",
            "_sha3",
            "_winapi",
            "telnetlib",
            "pyexpat",
            "_lzma",
            "http",
            "poplib",
            "tokenize",
            "_dbm",
            "_io",
            "linecache",
            "json",
            "faulthandler",
            "hmac",
            "aifc",
            "_csv",
            "_codecs_hk",
            "selectors",
            "_random",
            "_pickle",
            "_lsprof",
            "turtledemo",
            "cgitb",
            "_sitebuiltins",
            "binhex",
            "fnmatch",
            "sysconfig",
            "datetime",
            "quopri",
            "copyreg",
            "_pydecimal",
            "pty",
            "stringprep",
            "bisect",
            "_abc",
            "_codecs_jp",
            "_md5",
            "errno",
            "compileall",
            "_threading_local",
            "dbm",
            "builtins",
            "difflib",
            "imghdr",
            "__future__",
            "_statistics",
            "getopt",
            "xmlrpc",
            "_sqlite3",
            "_sha1",
            "shelve",
            "_posixshmem",
            "struct",
            "timeit",
            "ensurepip",
            "pathlib",
            "ctypes",
            "_multiprocessing",
            "tty",
            "_weakrefset",
            "sqlite3",
            "tracemalloc",
            "venv",
            "unittest",
            "_blake2",
            "mailbox",
            "resource",
            "shutil",
            "winreg",
            "_opcode",
            "_codecs_tw",
            "_operator",
            "imp",
            "_string",
            "os",
            "opcode",
            "_zoneinfo",
            "_posixsubprocess",
            "copy",
            "symtable",
            "itertools",
            "sre_parse",
            "_bisect",
            "_imp",
            "re",
            "ast",
            "zlib",
            "fractions",
            "pickle",
            "profile",
            "sys",
            "ssl",
            "cgi",
            "enum",
            "modulefinder",
            "py_compile",
            "_curses",
            "_functools",
            "cmath",
            "_crypt",
            "contextvars",
            "math",
            "uuid",
            "argparse",
            "_frozen_importlib",
            "inspect",
            "posix",
            "statistics",
            "marshal",
            "nis",
            "_bz2",
            "pipes",
            "socketserver",
            "pstats",
            "site",
            "trace",
            "lib2to3",
            "zipapp",
            "runpy",
            "sre_compile",
            "time",
            "pprint",
            "base64",
            "_stat",
            "_ast",
            "pdb",
            "_markupbase",
            "_bootsubprocess",
            "_collections",
            "_sre",
            "msilib",
            "crypt",
            "gettext",
            "mimetypes",
            "_overlapped",
            "asyncore",
            "zipimport",
            "chunk",
            "atexit",
            "graphlib",
            "_multibytecodec",
            "gzip",
            "io",
            "logging",
            "nntplib",
            "genericpath",
            "syslog",
            "token",
            "_msi",
            "idlelib",
            "_hashlib",
            "threading",
            "select",
            "doctest",
            "getpass",
            "_sha256",
            "importlib",
            "_tracemalloc",
            "multiprocessing",
            "calendar",
            "_codecs_cn",
            "_tkinter",
            "_uuid",
            "socket",
            "antigravity",
            "string",
            "_locale",
            "_thread",
            "grp",
            "this",
            "zoneinfo",
            "abc",
            "operator",
            "colorsys",
            "tabnanny",
            "_weakref",
            "imaplib",
            "concurrent",
            "subprocess",
            "_compression",
            "pyclbr",
            "tarfile",
            "numbers",
            "queue",
            "posixpath",
            "smtpd",
            "webbrowser",
            "asynchat",
            "weakref",
            "filecmp",
            "decimal",
            "_py_abc",
            "collections",
            "tempfile",
            "_collections_abc",
            "sched",
            "locale",
            "secrets",
            "msvcrt",
            "asyncio",
            "array",
            "_codecs_kr",
            "_scproxy",
            "_strptime",
            "heapq",
            "_socket",
            "sndhdr",
            "types",
            "nt",
            "_datetime",
            "shlex",
            "tkinter",
            "curses",
            "encodings",
            "pickletools",
            "html",
            "_codecs",
            "codeop",
            "_ctypes",
            "bz2",
            "contextlib",
            "platform",
            "termios",
            "_asyncio",
            "ftplib",
            "pydoc_data",
            "_contextvars",
            "codecs",
            "traceback",
            "pydoc",
            "fileinput",
            "ossaudiodev",
            "urllib",
            "csv",
            "sunau",
            "_curses_panel",
            "wave",
            "mmap",
            "warnings",
            "functools",
            "ipaddress",
            "nturl2path",
            "optparse",
            "_queue",
            "turtle",
            "spwd",
            "stat",
            "configparser",
            "_warnings",
            "bdb",
            "_osx_support",
            "typing",
            "zipfile",
            "glob",
            "random",
            "smtplib",
            "plistlib",
            "hashlib",
            "_struct",
        ]
        package_list = [
            package for package in package_list if package not in py10stdlib
        ]
        return package_list
