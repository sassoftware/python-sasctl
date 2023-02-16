# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Standard Library Imports
import ast
import importlib
import json
import math
import pickle
import pickletools
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import List, Union

# Third Party Imports
import pandas as pd

from ..core import current_session
from ..utils.decorators import deprecated

# TODO: file writing outputs should be dependent upon use in Jupyter notebook
# TODO: add converter for any type of dataset (list, dataframe, numpy array)

# Constants
INPUT = "inputVar.json"
OUTPUT = "outputVar.json"
PROP = "ModelProperties.json"
META = "fileMetadata.json"
FITSTAT = "dmcas_fitstat.json"
ROC = "dmcas_roc.json"
LIFT = "dmcas_lift.json"


def _flatten(nested_list):
    """
    Flatten a nested list.

    Flattens a nested list, while controlling for str values in list, such that the
    str values are not expanded into a list of single characters.

    Parameters
    ----------
    nested_list : list
        A nested list of strings.

    Yields
    ------
    list
        A flattened list of strings.
    """
    for item in nested_list:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from _flatten(item)
        else:
            yield item


class JSONFiles:
    @classmethod
    def write_var_json(
        cls,
        input_data: Union[dict, pd.DataFrame, pd.Series],
        is_input=True,
        json_path=None,
    ):
        """
        Writes a variable descriptor JSON file for input or output variables,
        based on input data containing predictor and prediction columns.

        If a path is provided, this function creates a JSON file named either
        inputVar.json or outputVar.json based on argument inputs. Otherwise, a dict
        is returned with the key-value pair representing the file name and json dump
        respectively.

        Parameters
        ----------
        input_data : dataframe or list of dicts
            Input dataframe containing the training data set in a pandas.Dataframe
            format. Columns are used to define predictor and prediction variables
            (ambiguously named "predict"). Providing a list of dict objects signals
            that the model files are being created from an MLFlow model.
        is_input : bool
            Boolean flag to check if generating the input or output variable JSON.
        json_path : string or Path, optional
            File location for the output JSON file. Default is None.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the file name and json
            dump respectively.
        """
        # MLFlow model handling
        if isinstance(input_data, list):
            dict_list = cls.generate_mlflow_variable_properties(input_data)
        # Normal model handling
        else:
            dict_list = cls.generate_variable_properties(input_data)
        if json_path:
            if is_input:
                file_name = INPUT
            else:
                file_name = OUTPUT

            with open(Path(json_path) / file_name, "w") as json_file:
                json_file.write(json.dumps(dict_list, indent=4))
            print(
                f"{file_name} was successfully written and saved to "
                f"{Path(json_path) / file_name}"
            )
        else:
            if is_input:
                return {INPUT: json.dumps(dict_list)}
            else:
                return {OUTPUT: json.dumps(dict_list)}

    @staticmethod
    def generate_variable_properties(input_data):
        """
        Generate a list of dictionaries of variable properties given an input dataframe.

        Parameters
        ----------
        input_data : pandas Dataframe or Series
            Dataset for either the input or output example data for the model.

        Returns
        -------
        dict_list : list of dicts
            List of dictionaries containing the variable properties.
        """
        # Check if input_data is a Series or DataFrame
        try:
            predict_names = input_data.columns.values.tolist()
            is_series = False
        except AttributeError:
            predict_names = [input_data.name]
            is_series = True

        dict_list = []
        # Loop through variables to determine properties
        for name in predict_names:
            if is_series:
                predict = input_data
            else:
                predict = input_data[name]
            first_row = predict.loc[predict.first_valid_index()]
            data_type = predict.dtypes.name
            is_str = type(first_row) is str

            var_dict = {"name": name}
            if is_str:
                var_dict.update(
                    {
                        "level": "nominal",
                        "type": "string",
                        "length": predict.str.len().max(),
                    }
                )
            else:
                if data_type == "category":
                    var_dict.update({"level": "nominal"})
                else:
                    var_dict.update({"level": "interval"})
                var_dict.update({"type": "decimal", "length": 8})

            dict_list.append(var_dict)

        return dict_list

    @classmethod
    def generate_mlflow_variable_properties(cls, input_data: list):
        """
        Create a list of dictionaries containing the variable properties found in the
        MLModel file for MLFlow model runs.

        Parameters
        ----------
        input_data : list of dicts
            Data pulled from the MLModel file by mlflow_model.py.

        Returns
        -------
        dict_list : list of dicts
            List of dictionaries containing the variable properties.
        """
        # Handle MLFlow models with different `var` formatting
        try:
            predict_names = [var["name"] for var in input_data]
        except KeyError:
            predict_names = [var["type"] for var in input_data]

        dict_list = []
        for i, name in enumerate(predict_names):
            is_str = cls.check_if_string(input_data[i])

            var_dict = {"name": name}
            if is_str:
                var_dict.update({"level": "nominal", "type": "string", "length": 8})
            else:
                var_dict.update({"level": "interval", "type": "decimal", "length": 8})
            dict_list.append(var_dict)

        return dict_list

    @staticmethod
    def check_if_string(data: dict):
        """
        Determine if an MLFlow variable in data is a string type.

        Parameters
        ----------
        data : dict
            Dictionary representation of a single variable from an MLFlow model.

        Returns
        -------
        bool
            True if the variable is a string. False otherwise.
        """
        if data["type"] == "string":
            return True
        elif data["type"] == "tensor":
            if data["tensor-spec"]["dtype"] in "string":
                return True
            else:
                return False
        else:
            return False

    # TODO: multiclass support -> target_variable to target_variables
    @staticmethod
    def write_model_properties_json(
        model_name,
        target_variable,
        target_event,
        num_target_categories,
        event_prob_var=None,
        json_path=None,
        model_desc=None,
        model_function=None,
        model_type=None,
        modeler=None,
        train_table=None,
    ):
        """
        Writes a JSON file containing SAS Model Manager model properties.

        The JSON file format is required by the SAS Model Repository API service and
        only event_prob_var can be 'None'. If a json_path is supplied, this function
        outputs a JSON file named "ModelProperties.json". Otherwise, a dict is returned.

        Parameters
        ----------
        model_name : string
            User-defined model name. This value is overwritten by SAS Model Manager
            based on the name of the zip file used for importing the model.
        target_variable : string
            Target variable to be predicted by the model.
        target_event : string
            Model target event. For example: 1 for a binary event.
        num_target_categories : int
            Number of possible target categories. For example: 2 for a binary event.
        event_prob_var : string, optional
            User-provided output event probability variable. This value should match the
            value in outputVar.json. Default is "P_" + target_variable + target_event.
        json_path : string or Path, optional
            Path for an output ModelProperties.json file to be generated. If no value
            is supplied a dict is returned instead. Default is None.
        model_desc : string, optional
            User-defined model description. Default is an empty string.
        model_function : string, optional
            User-defined model function. Default is an empty string.
        model_type : string, optional
            User-defined model type. Default is an empty string.
        modeler : string, optional
            User-defined value for the name of the modeler. Default is an empty string.
        train_table : string, optional
            The path to the model's training table within SAS Viya. Default is an empty
            string.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the file name and json
            dump respectively.
        """
        if model_desc:
            # Check if model description is smaller than the 1024-character limit
            if len(model_desc) > 1024:
                model_desc = model_desc[:1024]
                warnings.warn(
                    "WARNING: The provided model description was truncated to 1024 "
                    "characters. "
                )
        else:
            model_desc = ""

        if num_target_categories > 2:
            target_level = "INTERVAL"
        else:
            target_level = "BINARY"

        if event_prob_var is None:
            try:
                event_prob_var = "P_" + target_variable + target_event
            except TypeError:
                event_prob_var = None

        python_version = sys.version.split(" ", 1)[0]

        output_json = {
            "name": model_name,
            "description": model_desc,
            "function": model_function,
            "scoreCodeType": "python",
            "trainTable": train_table if train_table else "",
            "trainCodeType": "Python",
            "algorithm": model_type if model_type else "",
            "target_variable": target_variable,
            "target_event": target_event,
            "target_level": target_level,
            "event_prob_var": event_prob_var if event_prob_var else "",
            "modeler": modeler if modeler else "",
            "tool": "Python 3",
            "toolVersion": python_version,
        }

        if json_path:
            with open(Path(json_path) / PROP, "w") as json_file:
                json_file.write(json.dumps(output_json, indent=4))
            print(
                f"{PROP} was successfully written and saved to "
                f"{Path(json_path) / PROP}"
            )
        else:
            return {PROP: json.dumps(output_json)}

    @staticmethod
    def write_file_metadata_json(model_prefix, json_path=None, is_h2o_model=False):
        """
        Writes a file metadata JSON file pointing to all relevant files.

        This function outputs a JSON file named "fileMetadata.json".

        Parameters
        ----------
        model_prefix : string
            The variable for the model name that is used when naming model files. For
            example: hmeqClassTree + [Score.py | .pickle].
        json_path : string or Path, optional
            Path for an output ModelProperties.json file to be generated. If no value
            is supplied a dict is returned instead. Default is None.
        is_h2o_model : boolean, optional
            Sets whether the model metadata is associated with an H2O.ai model. If set
            as True, the MOJO model file will be set as a score resource. The default
            value is False.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the file name and json
            dump respectively.
        """
        dict_list = [
            {"role": "inputVariables", "name": INPUT},
            {"role": "outputVariables", "name": OUTPUT},
            {"role": "score", "name": model_prefix + "Score.py"},
        ]
        if is_h2o_model:
            dict_list.append({"role": "scoreResource", "name": model_prefix + ".mojo"})
        else:
            dict_list.append(
                {"role": "scoreResource", "name": model_prefix + ".pickle"}
            )

        if json_path:
            with open(Path(json_path) / META, "w") as json_file:
                json_file.write(json.dumps(dict_list, indent=4))
            print(
                f"{META} was successfully written and saved to "
                f"{Path(json_path) / META}"
            )
        else:
            return {META: json.dumps(dict_list, indent=4)}

    @classmethod
    def input_fit_statistics(
        cls,
        fitstat_df: pd.DataFrame = None,
        user_input: bool = False,
        tuple_list: List[tuple] = None,
        json_path: Union[str, Path] = None,
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

        The following are the base statistical parameters SAS Viya supports:
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
        fitstat_df : pandas Dataframe, optional
            Dataframe containing fitstat parameters and values. The default value is
            None.
        user_input : boolean, optional
            If true, prompt the user for more parameters. The default value is false.
        tuple_list : list of tuples, optional
            Input parameter tuples in the form of (parameterName, parameterValue,
            data_role). For example, a sample parameter call would be
            'NObs', 3488, or 'TRAIN'. Variable data_role is typically either TRAIN,
            TEST, or VALIDATE or 1, 2, 3 respectively. The default value is None.
        json_path : string or Path, optional
            Location for the output JSON file. The default value is None.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the file name and json
            dump respectively.
        """
        valid_params = [
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

        json_template_path = (
            Path(__file__).resolve().parent / "template_files/dmcas_fitstat.json"
        )
        json_dict = cls.read_json_file(json_template_path)

        data_map = [{}, {}, {}]
        for i in range(3):
            data_map[i] = json_dict["data"][i]

        if tuple_list:
            data_map = cls.add_tuple_to_fitstat(data_map, tuple_list, valid_params)

        if user_input:
            data_map = cls.user_input_fitstat(data_map, valid_params)

        if fitstat_df is not None:
            data_map = cls.add_df_to_fitstat(fitstat_df, data_map, valid_params)

        for i in range(3):
            json_dict["data"][i] = data_map[i]

        if json_path:
            with open(Path(json_path) / FITSTAT, "w") as json_file:
                json_file.write(json.dumps(json_dict, indent=4))
            print(
                f"{FITSTAT} was successfully written and saved to "
                f"{Path(json_path) / FITSTAT}"
            )
        else:
            return {FITSTAT: json.dumps(json_dict, indent=4)}

    @classmethod
    def add_tuple_to_fitstat(
        cls, data: List[dict], parameters: List[tuple], valid_params
    ):
        """
        Using tuples defined in input_fit_statistics, add them to the dmcas_fitstat json
        dictionary.

        Warnings are produced for invalid parameters found in the tuple.

        Parameters
        ----------
        data : list of dicts
            List of dicts for the data values of each parameter. Split into the three
            valid partitions (TRAIN, TEST, VALIDATE).
        parameters : list of tuples
            User-provided data for each parameter per partition provided.
        valid_params : list
            A list of valid parameters for dmcas_fitstat.json files.

        Returns
        -------
        list of dicts
            List of dicts with the tuple values inputted.

        Raises
        ------
        ValueError
            If an parameter within the tuple list is not a tuple or has a length
            different from the expected three.

        """
        for param in parameters:
            # Produce a warning or error for invalid parameter names or formatting
            if isinstance(param, tuple) and len(param) == 3:
                param_name = cls.format_parameter(param[0])
                if param_name not in valid_params:
                    warnings.warn(
                        f"WARNING: {param[0]} is not a valid parameter and has been "
                        f"ignored.",
                        category=UserWarning,
                    )
                    continue
                if isinstance(param[2], str):
                    data_role = cls.convert_data_role(param[2])
                else:
                    data_role = param[2]
                data[data_role - 1]["dataMap"][param_name] = param[1]
            elif not isinstance(param, tuple):
                raise ValueError(
                    f"Expected a tuple, but got {str(type(param))} instead."
                )
            elif len(param) != 3:
                raise ValueError(
                    f"Expected a tuple with three parameters, but instead got tuple "
                    f"with length {len(param)} "
                )
        return data

    @classmethod
    def user_input_fitstat(cls, data: List[dict], valid_params):
        """
        Prompt the user to enter parameters for dmcas_fitstat.json.

        Parameters
        ----------
        data : list of dicts
            List of dicts for the data values of each parameter. Split into the three
            valid partitions (TRAIN, TEST, VALIDATE).
        valid_params : list
            A list of valid parameters for dmcas_fitstat.json files.

        Returns
        -------
        list of dicts
            List of dicts with the user provided values inputted.
        """
        while True:
            input_param_name = input("What is the parameter name?\n")
            param_name = cls.format_parameter(input_param_name)
            if param_name not in valid_params:
                warnings.warn(
                    f"{input_param_name} is not a valid parameter.",
                    category=UserWarning,
                )
                if input("Would you like to input more parameters? (Y/N)") == "N":
                    break
                continue
            param_value = input("What is the parameter's value?\n")
            input_data_role = input(
                "Which data role is the parameter associated with?\n"
            )
            if isinstance(input_data_role, str):
                data_role = cls.convert_data_role(input_data_role)
            elif input_data_role in [1, 2, 3]:
                data_role = input_data_role
            else:
                warnings.warn(
                    f"{input_data_role} is not a valid role value. It should be either "
                    f"1, 2, or 3 or TRAIN, TEST, or VALIDATE respectively.",
                    category=UserWarning,
                )
                if input("Would you like to input more parameters? (Y/N)") == "N":
                    break
                continue
            data[data_role - 1]["dataMap"][param_name] = param_value

            if input("More parameters? (Y/N)") == "N":
                break
        return data

    @classmethod
    def add_df_to_fitstat(cls, df, data: List[dict], valid_params):
        """
        Add parameters from provided Dataframe to the fitstats dictionary.

        Parameters
        ----------
        df : pandas Dataframe
            Dataframe containing fitstat parameters and values.
        data : list of dicts
            List of dicts for the data values of each parameter. Split into the three
            valid partitions (TRAIN, TEST, VALIDATE).
        valid_params : list
            A list of valid parameters for dmcas_fitstat.json files.

        Returns
        -------
        list of dicts
            List of dicts with the user provided values inputted.
        """
        for i, row in enumerate(df.values):
            input_param_name, param_value, data_role = row
            param_name = cls.format_parameter(input_param_name)
            if param_name not in valid_params:
                warnings.warn(
                    f"{input_param_name} is not a valid parameter.",
                    category=UserWarning,
                )
                continue
            if isinstance(data_role, str):
                data_role = cls.convert_data_role(data_role)
            elif isinstance(data_role, int) and data_role not in [1, 2, 3]:
                warnings.warn(
                    f"{data_role} is not a valid role value. It should be either "
                    f"1, 2, or 3 or TRAIN, TEST, or VALIDATE respectively.",
                    category=UserWarning,
                )
                continue
            data[data_role - 1]["dataMap"][param_name] = param_value
        return data

    @classmethod
    def calculate_model_statistics(
        cls,
        target_value,
        prob_value=None,
        validate_data=None,
        train_data=None,
        test_data=None,
        json_path=None,
    ):
        """
        Calculates fit statistics (including ROC and Lift curves) from datasets and then
        either writes them to JSON files or returns them as a single dictionary.

        Calculations are performed using a call to SAS CAS via the swat package. An
        error will be raised if the swat package is not installed or if a connection to
        a SAS Viya system is not possible.

        Datasets must contain the actual and predicted values and may optionally contain
        the predicted probabilities. If no probabilities are provided, a dummy
        probability dataset is generated based on the predicted values and normalized by
        the target value. If a probability threshold value is not provided, the
        threshold value is set at 0.5.

        Datasets can be provided in the following forms, with the assumption that data
        is ordered as `actual`, `predict`, and `probability` respectively:
        * pandas dataframe: the actual and predicted values are their own columns
        * numpy array: the actual and predicted values are their own columns or rows and
        ordered such that the actual values come first and the predicted second
        * list: the actual and predicted values are their own indexed entry

        If a json_path is supplied, then this function outputs a set of JSON files named
        "dmcas_fitstat.json", "dmcas_roc.json", "dmcas_lift.json".

        Parameters
        ----------
        target_value : str, int, or float
            Target event value for model prediction events.
        prob_value : int or float, optional
            The threshold value for model predictions to indicate an event occurred. The
            default value is 0.5.
        validate_data : pandas DataFrame, list of lists, or numpy array, optional
            Dataset pertaining to the validation data. The default value is None.
        train_data : pandas DataFrame, list of lists, or numpy array, optional
            Dataset pertaining to the training data. The default value is None.
        test_data : pandas DataFrame, list of lists, or numpy array, optional
            Dataset pertaining to the test data. The default value is None.
        json_path : string or Path, optional
            Location for the output JSON files. The default value is None.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the files name and json
            dumps respectively.

        Raises
        ------
        RuntimeError
            If swat is not installed, this function cannot perform the necessary
            calculations.
        """
        try:
            sess = current_session()
            conn = sess.as_swat()
        except ImportError:
            raise RuntimeError(
                "The `swat` package is required to generate fit statistics, ROC, and "
                "Lift charts with the calculate_model_statistics function."
            )

        json_dict = [{}, {}, {}]
        for i, name in enumerate(["dmcas_fitstat", "dmcas_roc", "dmcas_lift"]):
            json_template_path = (
                Path(__file__).resolve().parent / f"template_files/{name}.json"
            )
            json_dict[i] = cls.read_json_file(json_template_path)

        conn.loadactionset(actionset="percentile")

        data_partition_exists = cls.check_for_data(validate_data, train_data, test_data)

        for i, (partition, data) in enumerate(
            zip(data_partition_exists, [validate_data, train_data, test_data])
        ):
            # If the data partition was not passed, skip to the next partition
            if not partition:
                continue

            data = cls.stat_dataset_to_dataframe(data, target_value)

            conn.upload(
                data,
                casout={"name": "assess_dataset", "replace": True, "caslib": "Public"},
            )

            conn.percentile.assess(
                table={"name": "assess_dataset", "caslib": "Public"},
                response="predict",
                pVar="predict_proba",
                event=str(target_value),
                pEvent=str(prob_value) if prob_value else str(0.5),
                inputs="actual",
                fitStatOut={"name": "FitStat", "replace": True, "caslib": "Public"},
                rocOut={"name": "ROC", "replace": True, "caslib": "Public"},
                casout={"name": "Lift", "replace": True, "caslib": "Public"},
            )

            fitstat_dict = (
                pd.DataFrame(conn.CASTable("FitStat", caslib="Public").to_frame())
                .transpose()
                .squeeze()
                .to_dict()
            )
            json_dict[0]["data"][i]["dataMap"].update(fitstat_dict)

            roc_df = pd.DataFrame(conn.CASTable("ROC", caslib="Public").to_frame())
            roc_dict = cls.apply_dataframe_to_json(json_dict[1]["data"], i, roc_df)
            for j in range(len(roc_dict)):
                json_dict[1]["data"][j].update(roc_dict[j])

            lift_df = pd.DataFrame(conn.CASTable("Lift", caslib="Public").to_frame())
            lift_dict = cls.apply_dataframe_to_json(json_dict[2]["data"], i, lift_df)
            for j in range(len(lift_dict)):
                json_dict[2]["data"][j].update(lift_dict[j])

        if json_path:
            for name in [FITSTAT, ROC, LIFT]:
                with open(Path(json_path) / name, "w") as json_file:
                    json_file.write(json.dumps(json_dict, indent=4))
                print(
                    f"{name} was successfully written and saved to "
                    f"{Path(json_path) / name}"
                )
        else:
            return {
                FITSTAT: json.dumps(json_dict[0], indent=4),
                ROC: json.dumps(json_dict[1], indent=4),
                LIFT: json.dumps(json_dict[2], indent=4),
            }

    @staticmethod
    def check_for_data(validate, train, test):
        """
        Check which datasets were provided and return a list of flags.

        Parameters
        ----------
        validate : pandas DataFrame, list of lists, or numpy array
            Dataset pertaining to the validation data.
        train : pandas DataFrame, list of lists, or numpy array
            Dataset pertaining to the training data.
        test : pandas DataFrame, list of lists, or numpy array
            Dataset pertaining to the test data.

        Returns
        -------
        data_partitions : list
            A list of flags indicating which partitions have datasets.

        Raises
        ------
        ValueError
            If no data is provided, raises an exception.
        """
        if all(data is None for data in (validate, train, test)):
            raise ValueError(
                "No data was provided. Please provide the actual and predicted values "
                "for at least one of the partitions (VALIDATE, TRAIN, or TEST)."
            )
        else:
            data_partitions = [
                1 if validate is not None else 0,
                1 if train is not None else 0,
                1 if test is not None else 0,
            ]
        return data_partitions

    @staticmethod
    def stat_dataset_to_dataframe(data, target_value=None):
        """
        Convert the user supplied statistical dataset from either a pandas DataFrame,
        list of lists, or numpy array to a DataFrame formatted for SAS CAS upload.

        If the prediction probabilities are not provided, the prediction data will be
        duplicated to allow for calculation of the fit statistics through CAS and then a
        binary filter is applied to the duplicate column based off of a provided target
        value. The data is assumed to be in the order of "actual", "predicted",
        "probability" respectively.

        Parameters
        ----------
        data : pandas DataFrame, list of lists, or numpy array
            Dataset representing the actual and predicted values of the model. May also
            include the prediction probabilities.
        target_value : str, int, or float, optional
            Target event value for model prediction events. Used for creating a binary
            probability column when no probability values are provided. Default is None.

        Returns
        -------
        data : pandas DataFrame
            Dataset formatted for SAS CAS upload.

        Raises
        ------
        ValueError
            Raised if an improper data format is provided.

        """
        # If numpy inputs are supplied, then assume numpy is installed
        try:
            # noinspection PyPackageRequirements
            import numpy as np
        except ImportError:
            np = None

        # Convert target_value to numeric for creating binary probabilities
        if isinstance(target_value, str):
            target_value = float(target_value)

        # Assume column order (actual, predicted, probability) per argument instructions
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 2:
                data.columns = ["actual", "predict"]
                data["predict_proba"] = data["predict"].gt(target_value).astype(int)
            elif len(data.columns) == 3:
                data.columns = ["actual", "predict", "predict_proba"]
        elif isinstance(data, list):
            if len(data) == 2:
                data = pd.DataFrame({"actual": data[0], "predict": data[1]})
                data["predict_proba"] = data["predict"].gt(target_value).astype(int)
            elif len(data) == 3:
                data = pd.DataFrame(
                    {
                        "actual": data[0],
                        "predict": data[1],
                        "predict_proba": data[2],
                    }
                )
        elif isinstance(data, np.ndarray):
            if len(data) == 2:
                data = pd.DataFrame({"actual": data[0, :], "predict": data[1, :]})
                data["predict_proba"] = data["predict"].gt(target_value).astype(int)
            elif len(data) == 3:
                data = pd.DataFrame(
                    {"actual": data[0], "predict": data[1], "predict_proba": data[2]}
                )
        else:
            raise ValueError(
                "Please provide the data in a list of lists, dataframe, or numpy array."
            )

        return data

    @staticmethod
    def apply_dataframe_to_json(json_dict, partition, stat_df):
        """
        Map the values of the ROC or Lift charts from SAS CAS to the dictionary
        representation of the respective json file.

        Parameters
        ----------
        json_dict : dict
            Dictionary representation of the ROC or Lift chart json file.
        partition : int
            Numerical representation of the data partition. Either 0, 1, or 2.
        stat_df : pandas DataFrame
            ROC or Lift DataFrame generated from the SAS CAS percentile action set.

        Returns
        -------
        json_dict : dict
            Dictionary representation of the ROC or Lift chart json file, with the
            values from the SAS CAS percentile action set added in.
        """
        for row_num in range(len(stat_df)):
            row_dict = stat_df.iloc[row_num].to_dict()
            json_dict[row_num + partition * len(stat_df)]["dataMap"].update(row_dict)
        return json_dict

    # noinspection PyCallingNonCallable,PyNestedDecorators
    @deprecated(
        "Please use the calculate_model_statistics method instead.",
        version="1.9",
        removed_in="1.10",
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

        nullJSONPath = (
            Path(__file__).resolve().parent / "template_files/dmcas_fitstat.json"
        )
        nullJSONDict = cls.read_json_file(nullJSONPath)

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

        with open(Path(jPath) / FITSTAT, "w") as jFile:
            json.dump(nullJSONDict, jFile, indent=4)
        print(
            f"{FITSTAT} was successfully written and saved to {Path(jPath) / FITSTAT}"
        )

    # noinspection PyCallingNonCallable,PyNestedDecorators
    @deprecated(
        "Please use the calculate_model_statistics method instead.",
        version="1.9",
        removed_in="1.10",
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

        nullJSONROCPath = (
            Path(__file__).resolve().parent / "template_files/dmcas_roc.json"
        )
        nullJSONROCDict = cls.read_json_file(nullJSONROCPath)

        nullJSONLiftPath = (
            Path(__file__).resolve().parent / "template_files/dmcas_lift.json"
        )
        nullJSONLiftDict = cls.read_json_file(nullJSONLiftPath)

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

        with open(Path(jPath) / ROC, "w") as jFile:
            json.dump(nullJSONROCDict, jFile, indent=4)
        print(f"{ROC} was successfully written and saved to {Path(jPath) / ROC}")

        with open(Path(jPath) / LIFT, "w") as jFile:
            json.dump(nullJSONLiftDict, jFile, indent=4)
        print(f"{LIFT} was successfully written and saved to {Path(jPath) / LIFT}")

    @staticmethod
    def read_json_file(path):
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
    def format_parameter(param_name):
        """
        Formats the parameter name to the JSON standard expected for dmcas_fitstat.json.

        Parameters
        ----------
        param_name : string
            Name of the parameter.

        Returns
        -------
        paramName : string
            Name of the parameter.
        """
        if not (param_name.startswith("_") and param_name.endswith("_")):
            if not param_name.startswith("_"):
                param_name = "_" + param_name
            if not param_name.endswith("_"):
                param_name = param_name + "_"

        return param_name

    @staticmethod
    def convert_data_role(data_role):
        """
        Converts the data role identifier from string to int or int to string.

        JSON file descriptors require the string, int, and formatted int. If the
        provided data role is not valid, defaults to TRAIN (1).

        Parameters
        ----------
        data_role : string or int
            Identifier of the data set's role; either TRAIN, TEST, or VALIDATE, or
            correspondingly 1, 2, or 3.

        Returns
        -------
        conversion : int or string
            Converted data role identifier.
        """
        if isinstance(data_role, int) or isinstance(data_role, float):
            data_role = int(data_role)
            if data_role == 1:
                conversion = "TRAIN"
            elif data_role == 2:
                conversion = "TEST"
            elif data_role == 3:
                conversion = "VALIDATE"
            else:
                conversion = "TRAIN"
        elif isinstance(data_role, str):
            if data_role.upper() == "TRAIN":
                conversion = 1
            elif data_role.upper() == "TEST":
                conversion = 2
            elif data_role.upper() == "VALIDATE":
                conversion = 3
            else:
                conversion = 1
        else:
            conversion = 1

        return conversion

    @classmethod
    def create_requirements_json(cls, model_path=Path.cwd(), output_path=None):
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

        When provided with an output_path argument, this function outputs a JSON file
        named "requirements.json". Otherwise, a list of dicts is returned.

        Parameters
        ----------
        model_path : str or Path, optional
            The path to a Python project, by default the current working directory.
        output_path : str or Path, optional
            The path for the output requirements.json file. Default is None.
        Returns
        -------
        list of dicts
            List of dictionary representations of the json file contents, split into
            each package and/or warning.
        """
        pickle_packages = []
        pickle_files = cls.get_pickle_file(model_path)
        for pickle_file in pickle_files:
            pickle_packages.append(cls.get_pickle_dependencies(pickle_file))

        code_dependencies = cls.get_code_dependencies(model_path)

        package_list = list(pickle_packages) + list(code_dependencies)
        package_list = list(set(list(_flatten(package_list))))
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
        if output_path:
            with open(  # skipcq: PTC-W6004
                Path(output_path) / "requirements.json", "w"
            ) as file:
                file.write(json.dumps(json_dicts, indent=4))
        else:
            return {"requirements.json": json.dumps(json_dicts)}

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
        model_path : string or Path, optional
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
        import_info = list(set(_flatten(import_info)))
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
        except ValueError:
            pass
        return modules

    @staticmethod
    def get_pickle_file(pickle_folder=Path.cwd()):
        """
        Given a file path, retrieve the pickle file(s).

        Parameters
        ----------
        pickle_folder : str or Path
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
        pickle_file : str or Path
            The file where you stored pickle data.

        Returns
        -------
        list
            A list of modules obtained from the pickle stream. Duplicates are removed
            and Python built-in modules are removed.
        """
        with open(pickle_file, "rb") as open_file:  # skipcq: PTC-W6004
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
