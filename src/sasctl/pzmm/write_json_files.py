# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Standard Library Imports
import ast
import importlib
import json

import pickle
import pickletools
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Generator, List, Optional, Type, Union

# Third Party Imports
import pandas as pd
from pandas import DataFrame, Series

# Package Imports
from sasctl.pzmm.write_score_code import ScoreCode as sc
from ..core import current_session
from ..utils.decorators import deprecated, experimental
from ..utils.misc import check_if_jupyter

try:
    # noinspection PyPackageRequirements
    import numpy as np

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

except ImportError:
    np = None

    class NpEncoder(json.JSONEncoder):
        pass


# TODO: add converter for any type of dataset (list, dataframe, numpy array)

# Constants
INPUT = "inputVar.json"
OUTPUT = "outputVar.json"
PROP = "ModelProperties.json"
META = "fileMetadata.json"
FITSTAT = "dmcas_fitstat.json"
ROC = "dmcas_roc.json"
LIFT = "dmcas_lift.json"
MAXDIFFERENCES = "maxDifferences.json"
GROUPMETRICS = "groupMetrics.json"
VARIMPORTANCES = "dmcas_relativeimportance.json"
MISC = "dmcas_misc.json"


def _flatten(nested_list: Iterable) -> Generator[Any, None, None]:
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
    notebook_output: bool = check_if_jupyter()
    valid_params: List[str] = [
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

    @classmethod
    def write_var_json(
        cls,
        input_data: Union[dict, DataFrame, Series],
        is_input: Optional[bool] = True,
        json_path: Union[str, Path, None] = None,
    ) -> Union[dict, None]:
        """
        Writes a variable descriptor JSON file for input or output variables,
        based on input data containing predictor and prediction columns.

        If a path is provided, this function creates a JSON file named either
        inputVar.json or outputVar.json based on argument inputs. Otherwise, a dict
        is returned with the key-value pair representing the file name and json dump
        respectively.

        Parameters
        ----------
        input_data : pandas.DataFrame, pandas.Series, or list of dict
            Input dataframe containing the training data set in a pandas.Dataframe
            format. Columns are used to define predictor and prediction variables
            (ambiguously named "predict"). Providing a list of dict objects signals
            that the model files are being created from an MLFlow model.
        is_input : bool, optional
            Boolean flag to check if generating the input or output variable JSON. The
            default value is True.
        json_path : str or pathlib.Path, optional
            File location for the output JSON file. The default value is None.

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
                json_file.write(json.dumps(dict_list, indent=4, cls=NpEncoder))
            if cls.notebook_output:
                print(
                    f"{file_name} was successfully written and saved to "
                    f"{Path(json_path) / file_name}"
                )
        else:
            if is_input:
                return {INPUT: json.dumps(dict_list, indent=4, cls=NpEncoder)}
            else:
                return {OUTPUT: json.dumps(dict_list, indent=4, cls=NpEncoder)}

    @staticmethod
    def generate_variable_properties(
        input_data: Union[DataFrame, Series],
    ) -> List[dict]:
        """
        Generate a list of dictionaries of variable properties given an input dataframe.

        Parameters
        ----------
        input_data : pandas.DataFrame or pandas.Series
            Dataset for either the input or output example data for the model.

        Returns
        -------
        dict_list : list of dict
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
                        "length": int(predict.str.len().max()),
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
    def generate_mlflow_variable_properties(cls, input_data: list) -> List[dict]:
        """
        Create a list of dictionaries containing the variable properties found in the
        MLModel file for MLFlow model runs.

        Parameters
        ----------
        input_data : list of dict
            Data pulled from the MLModel file by mlflow_model.py.

        Returns
        -------
        dict_list : list of dict
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
    def check_if_string(data: dict) -> bool:
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

    @classmethod
    def write_model_properties_json(
        cls,
        model_name: str,
        target_variable: str,
        target_values: Optional[List[Any]] = None,
        json_path: Union[str, Path, None] = None,
        model_desc: Optional[str] = None,
        model_algorithm: Optional[str] = None,
        model_function: Optional[str] = None,
        modeler: Optional[str] = None,
        train_table: Optional[str] = None,
        properties: Optional[List[dict]] = None,
    ) -> Union[dict, None]:
        """
        Writes a JSON file containing SAS Model Manager model properties.

        Property values for multiclass models are not supported on a model-level in SAS
        Model Manager. If these values are detected, they will be supplied as custom
        user properties.

        If a json_path is supplied, this function outputs a JSON file named
        "ModelProperties.json". Otherwise, a dict is returned.

        Parameters
        ----------
        model_name : str
            User-defined model name. This value is overwritten by SAS Model Manager
            based on the name of the zip file used for importing the model.
        target_variable : str
            Target variable to be predicted by the model.
        target_values : list, optional
            Model target event(s). Providing no target values indicates the model is a
            regression model. Providing 2 target values indicates the model is a binary
            classification model. Providing > 2 target values will supply the values
            for the different target events as a custom property. An error is raised if
            only 1 target value is supplied. The default value is None.
        json_path : str or pathlib.Path, optional
            Path for an output ModelProperties.json file to be generated. If no value
            is supplied a dict is returned instead. The default value is None.
        model_desc : str, optional
            User-defined model description. The default value is an empty string.
        model_algorithm : str, optional
            User-defined model algorithm name. The default value is an empty string.
        model_function : str, optional
            User-defined model function name. The default value is an empty string.
        modeler : str, optional
            User-defined value for the name of the modeler. The default value is an
            empty string.
        train_table : str, optional
            The path to the model's training table within SAS Viya. The default value is
            an empty string.
        properties : list of dict, optional
            List of custom properties to be shown in the user-defined properties section
            of the model in SAS Model Manager. Dict entries should contain the `name`,
            `value`, and `type` keys. The default value is an empty list.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the file name and json
            dump respectively.
        """
        if properties is None:
            properties = []

        if model_desc:
            # Check if model description is smaller than the 1024-character limit
            if len(model_desc) > 1024:
                model_desc = model_desc[:1024]
                warnings.warn(
                    "WARNING: The provided model description was truncated to 1024 "
                    "characters."
                )

        if not target_values:
            model_function = model_function if model_function else "prediction"
            target_level = "Interval"
            target_event = ""
            event_prob_var = ""
        elif isinstance(target_values, list) and len(target_values) == 2:
            model_function = model_function if model_function else "classification"
            target_level = "Binary"
            target_event = str(target_values[0])
            event_prob_var = f"P_{target_values[0]}"
        elif isinstance(target_values, list) and len(target_values) > 2:
            model_function = model_function if model_function else "classification"
            target_level = "Nominal"
            target_event = ""
            event_prob_var = ""
            targets = [str(x) for x in target_values]
            properties.append(
                {
                    "name": "multiclass_target_events",
                    "value": ", ".join(targets),
                    "type": "string",
                }
            )
            prob_targets = ["P_" + str(x) for x in target_values]
            properties.append(
                {
                    "name": "multiclass_proba_variables",
                    "value": ", ".join(prob_targets),
                    "type": "string",
                }
            )
        else:
            raise ValueError(
                "Please provide all possible values for the target variable, including"
                " a no-event value."
            )

        truncated_properties = []
        for prop in properties:
            prop = cls.truncate_properties(prop)
            truncated_properties.append(prop)

        python_version = sys.version.split(" ", 1)[0]

        output_json = {
            "name": model_name,
            "description": model_desc if model_desc else "",
            "scoreCodeType": "python",
            "trainTable": train_table if train_table else "",
            "trainCodeType": "Python",
            "algorithm": model_algorithm if model_algorithm else "",
            "function": model_function if model_function else "",
            "targetVariable": target_variable if target_variable else "",
            "targetEvent": target_event if target_event else "",
            "targetLevel": target_level if target_level else "",
            "eventProbVar": event_prob_var if event_prob_var else "",
            "modeler": modeler if modeler else "",
            "tool": "Python 3",
            "toolVersion": python_version,
            "properties": truncated_properties,
        }

        if json_path:
            with open(Path(json_path) / PROP, "w") as json_file:
                json_file.write(json.dumps(output_json, indent=4))
            if cls.notebook_output:
                print(
                    f"{PROP} was successfully written and saved to "
                    f"{Path(json_path) / PROP}"
                )
        else:
            return {PROP: json.dumps(output_json)}

    @staticmethod
    def truncate_properties(prop: dict) -> dict:
        """
        Check custom properties for values larger than SAS Model Manager expects.

        Property names cannot be larger than 60 characters. Property values cannot be
        larger than 512 characters.

        Parameters
        ----------
        prop : dict
            Key-value pair representing the property name and value.

        Returns
        -------
        prop : dict
            Key-value pair, which was truncated as needed by SAS Model Manager.
        """
        prop_key, prop_value = list(prop.items())[0]

        if len(prop_key) > 60:
            warnings.warn(
                f"WARNING: The property name {prop_key} was truncated to 60 "
                f"characters."
            )
            truncated_name = prop_key[:60]
            prop[truncated_name] = prop.pop(prop_key)
            prop_key = truncated_name

        if len(prop_value) > 512:
            warnings.warn(
                f"WARNING: The property value {prop_value} was truncated to 512 "
                f"characters."
            )
            truncated_value = prop_value[:512]
            prop.update({prop_key: truncated_value})

        return prop

    @classmethod
    def write_file_metadata_json(
        cls,
        model_prefix: str,
        json_path: Union[str, Path, None] = None,
        is_h2o_model: Optional[bool] = False,
        is_tf_keras_model: Optional[bool] = False,
    ) -> Union[dict, None]:
        """
        Writes a file metadata JSON file pointing to all relevant files.

        This function outputs a JSON file named "fileMetadata.json".

        Parameters
        ----------
        model_prefix : str
            The variable for the model name that is used when naming model files. For
            example: hmeqClassTree + [Score.py | .pickle].
        json_path : str or pathlib.Path, optional
            Path for an output ModelProperties.json file to be generated. If no value
            is supplied a dict is returned instead. The default value is None.
        is_h2o_model : bool, optional
            Sets whether the model metadata is associated with an H2O.ai model. If set
            as True, the MOJO model file will be set as a score resource. The default
            value is False.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the file name and json
            dump respectively.
        """

        from .write_score_code import ScoreCode

        sanitized_prefix = ScoreCode.sanitize_model_prefix(model_prefix)

        dict_list = [
            {"role": "inputVariables", "name": INPUT},
            {"role": "outputVariables", "name": OUTPUT},
            {"role": "score", "name": f"score_{sanitized_prefix}.py"},
        ]
        if is_h2o_model:
            dict_list.append(
                {"role": "scoreResource", "name": sanitized_prefix + ".mojo"}
            )
        elif is_tf_keras_model:
            dict_list.append(
                {"role": "scoreResource", "name": sanitized_prefix + ".h5"}
            )
        else:
            dict_list.append(
                {"role": "scoreResource", "name": sanitized_prefix + ".pickle"}
            )

        if json_path:
            with open(Path(json_path) / META, "w") as json_file:
                json_file.write(json.dumps(dict_list, indent=4))
            if cls.notebook_output:
                print(
                    f"{META} was successfully written and saved to "
                    f"{Path(json_path) / META}"
                )
        else:
            return {META: json.dumps(dict_list, indent=4)}

    @classmethod
    def input_fit_statistics(
        cls,
        fitstat_df: Optional[DataFrame] = None,
        user_input: Optional[bool] = False,
        tuple_list: Optional[List[tuple]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> Union[dict, None]:
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
        fitstat_df : pandas.DataFrame, optional
            Dataframe containing fitstat parameters and values. The default value is
            None.
        user_input : bool, optional
            If true, prompt the user for more parameters. The default value is false.
        tuple_list : list of tuple, optional
            Input parameter tuples in the form of (parameterName, parameterValue,
            data_role). For example, a sample parameter call would be
            'NObs', 3488, or 'TRAIN'. Variable data_role is typically either TRAIN,
            TEST, or VALIDATE or 1, 2, 3 respectively. The default value is None.
        json_path : str or pathlib.Path, optional
            Location for the output JSON file. The default value is None.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the file name and json
            dump respectively.
        """
        json_template_path = (
            Path(__file__).resolve().parent / "template_files/dmcas_fitstat.json"
        )
        json_dict = cls.read_json_file(json_template_path)

        data_map = [{}, {}, {}]
        for i in range(3):
            data_map[i] = json_dict["data"][i]

        if tuple_list:
            data_map = cls.add_tuple_to_fitstat(data_map, tuple_list)

        if user_input:
            data_map = cls.user_input_fitstat(data_map)

        if fitstat_df is not None:
            data_map = cls.add_df_to_fitstat(fitstat_df, data_map)

        for i in range(3):
            json_dict["data"][i] = data_map[i]

        if json_path:
            with open(Path(json_path) / FITSTAT, "w") as json_file:
                json_file.write(json.dumps(json_dict, indent=4, cls=NpEncoder))
            if cls.notebook_output:
                print(
                    f"{FITSTAT} was successfully written and saved to "
                    f"{Path(json_path) / FITSTAT}"
                )
        else:
            return {FITSTAT: json.dumps(json_dict, indent=4, cls=NpEncoder)}

    @classmethod
    def add_tuple_to_fitstat(
        cls, data: List[dict], parameters: List[tuple]
    ) -> List[dict]:
        """
        Using tuples defined in input_fit_statistics, add them to the dmcas_fitstat json
        dictionary.

        Warnings are produced for invalid parameters found in the tuple.

        Parameters
        ----------
        data : list of dict
            List of dicts for the data values of each parameter. Split into the three
            valid partitions (TRAIN, TEST, VALIDATE).
        parameters : list of tuple
            User-provided data for each parameter per partition provided.

        Returns
        -------
        list of dict
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
                if param_name not in cls.valid_params:
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
    def user_input_fitstat(cls, data: List[dict]) -> List[dict]:
        """
        Prompt the user to enter parameters for dmcas_fitstat.json.

        Parameters
        ----------
        data : list of dict
            List of dicts for the data values of each parameter. Split into the three
            valid partitions (TRAIN, TEST, VALIDATE).

        Returns
        -------
        list of dict
            List of dicts with the user provided values inputted.
        """
        while True:
            input_param_name = input("What is the parameter name?\n")
            param_name = cls.format_parameter(input_param_name)
            if param_name not in cls.valid_params:
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
    def add_df_to_fitstat(cls, df: DataFrame, data: List[dict]) -> List[dict]:
        """
        Add parameters from provided DataFrame to the fitstats dictionary.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing fitstat parameters and values.
        data : list of dict
            List of dicts for the data values of each parameter. Split into the three
            valid partitions (TRAIN, TEST, VALIDATE).

        Returns
        -------
        list of dict
            List of dicts with the user provided values inputted.
        """
        for i, row in enumerate(df.values):
            input_param_name, param_value, data_role = row
            param_name = cls.format_parameter(input_param_name)
            if param_name not in cls.valid_params:
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

    # TODO: Add unit/integration tests
    @classmethod
    @experimental
    def assess_model_bias(
        cls,
        score_table: DataFrame,
        sensitive_values: Union[str, List[str]],
        actual_values: str,
        pred_values: str = None,
        prob_values: List[str] = None,
        levels: List[str] = None,
        json_path: Union[str, Path, None] = None,
        cutoff: float = 0.5,
        datarole: str = "TEST",
        return_dataframes: bool = False,
    ) -> Union[dict, None]:
        """
        Calculates model bias metrics for sensitive variables and dumps metrics into SAS Viya readable JSON Files. This
        function works for regression and binary classification problems.

        Parameters
        ----------
        score_table : pandas.DataFrame
            Data structure containing actual values, predicted or predicted probability values, and sensitive variable
            values. All columns in the score table must have valid variable names.
        sensitive_values : str or list of str
            Sensitive variable name or names in score_table. The variable name must follow SAS naming conventions (no
            spaces and the name cannot begin with a number or symbol).
        actual_values : str
            Variable name containing the actual values in score_table. The variable name must follow SAS naming
            conventions (no spaces and the name cannot begin with a number or symbol).
        pred_values : str
            Required for regression problems, otherwise not used
            Variable name containing the predicted values in score_table. The variable name must follow SAS naming
            conventions (no spaces and the name cannot begin with a number or symbol).Required for regression problems.
            The default value is None.
        prob_values : list of str
            Required for classification problems, otherwise not used
            A list of variable names containing the predicted probability values in the score table. The first element
            should represent the predicted probability of the target class. Required for classification problems. Default
            is None.
        levels : list of str, list of int, or list of bool
            Required for classification problems, otherwise not used
            List of classes of a nominal target in the order they were passed in prob_values. Levels must be passed as a
            string. Default is None.
        json_path : str or pathlib.Path, optional
            Location for the output JSON files. If a path is passed, the json files will populate in the directory and
            the function will return None, unless return_dataframes is True. Otherwise, the function will return the json
            strings in a dictionary (dict["maxDifferences.json"] and dict["groupMetrics.json"]). The default value is
            None.
        cutoff : float, optional
            Cutoff value for confusion matrix. Default is 0.5.
        datarole : str, optional
            The data being used to assess bias (i.e. 'TEST', 'VALIDATION', etc.). Default is 'TEST.'
        return_dataframes : bool, optional
            If true, the function returns the pandas data frames used to create the JSON files and a table for bias
            metrics. If a JSON path is passed, then the function will return a dictionary that only includes the data
            frames (dict["maxDifferencesData"], dict["groupMetricData"], and dict["biasMetricsData"]). If a JSON path is
            not passed, the function will return a dictionary with the three tables  and the two JSON strings
            (dict["maxDifferences.json"] and dict["groupMetrics.json"]). The default value is False.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the files name and json
            dumps respectively.

        Raises
        ------
        RuntimeError
            If :mod:`swat` is not installed, this function cannot perform the necessary
            calculations.

        ValueError
            This function requires pred_values OR (regression) or prob_values AND levels (classification) to be passed.

            Variable names must follow SAS naming conventions (no spaces or names that begin with a number or symbol).
        """
        try:
            sess = current_session()
            conn = sess.as_swat()
        except ImportError:
            raise RuntimeError(
                "The `swat` package is required to generate fit statistics, ROC, and Lift charts with the "
                "calculate_model_statistics function."
            )

        variables = score_table.columns
        sc._check_for_invalid_variable_names(variables)

        if pred_values is None and prob_values is None:
            raise ValueError(
                "A value for pred_values (regression) or prob_values (classification) must be passed."
            )

        # if it's a classification problem
        if prob_values is not None:
            if levels is None:
                raise ValueError(
                    "Levels of the target variable must be passed for classification problems. The levels should be "
                    "ordered in the same way that the predicted probability variables are ordered."
                )
            score_table[actual_values] = score_table[actual_values].astype(str)

        if isinstance(sensitive_values, str):
            sensitive_values = [sensitive_values]

        # upload properly formatted score table to CAS
        conn.upload(score_table, casout=dict(name="score_table"))

        conn.loadactionset("fairaitools")
        maxdiff_dfs = []
        groupmetrics_dfs = []
        biasmetrics_dfs = []

        for x in sensitive_values:
            # run assessBias, if levels=None then assessBias treats the input like a regression problem
            tables = conn.fairaitools.assessbias(
                modelTableType="None",
                predictedVariables=(
                    pred_values if pred_values is not None else prob_values
                ),
                response=actual_values,
                responseLevels=levels,
                sensitiveVariable=x,
                cutoff=cutoff,
                table="score_table",
            )

            # get maxdiff table, append to list
            maxdiff = pd.DataFrame(tables["MaxDifferences"])
            # adding variable to table
            maxdiff["_VARIABLE_"] = x
            maxdiff_dfs.append(maxdiff)

            # get group metrics table, append to list
            group_metrics = pd.DataFrame(tables["GroupMetrics"])
            group_metrics["_VARIABLE_"] = x
            groupmetrics_dfs.append(group_metrics)

            # get bis metrics table if they want to return it
            if return_dataframes:
                bias_metrics = pd.DataFrame(tables["BiasMetrics"])
                bias_metrics["_VARIABLE_"] = x
                biasmetrics_dfs.append(bias_metrics)

        # overall formatting
        group_metrics = cls.format_group_metrics(
            groupmetrics_dfs=groupmetrics_dfs,
            prob_values=prob_values,
            pred_values=pred_values,
            datarole=datarole,
        )

        max_differences = cls.format_max_differences(
            maxdiff_dfs=maxdiff_dfs, datarole=datarole
        )

        # getting json files
        json_files = cls.bias_dataframes_to_json(
            groupmetrics=group_metrics,
            maxdifference=max_differences,
            n_sensitivevariables=len(sensitive_values),
            actual_values=actual_values,
            prob_values=prob_values,
            levels=levels,
            pred_values=pred_values,
            json_path=json_path,
        )

        if return_dataframes:
            bias_metrics = pd.concat(biasmetrics_dfs)
            df_dict = {
                "maxDifferencesData": max_differences,
                "groupMetricsData": group_metrics,
                "biasMetricsData": bias_metrics,
            }

            if json_files is None:
                return df_dict

            json_files.update(df_dict)

        return json_files

    @staticmethod
    def format_max_differences(
        maxdiff_dfs: List[DataFrame], datarole: str = "TEST"
    ) -> DataFrame:
        """
        Converts a list of max differences DataFrames into a singular DataFrame

        Parameters
        ----------
        maxdiff_dfs: list of pandas.DataFrame
            A list of max_differences DataFrames returned by CAS
        datarole : str, optional
            The data being used to assess bias (i.e. 'TEST', 'VALIDATION', etc.). Default is 'TEST.'

        Returns
        -------
        pandas.DataFrame
            A singluar DataFrame containing all max differences data
        """
        maxdiff_df = pd.concat(maxdiff_dfs)
        maxdiff_df = maxdiff_df.rename(
            columns={"Value": "maxdiff", "Base": "BASE", "Compare": "COMPARE"}
        )
        maxdiff_df["maxdiff"] = maxdiff_df["maxdiff"].apply(str)

        maxdiff_df["VLABEL"] = ""
        maxdiff_df["_DATAROLE_"] = datarole

        maxdiff_df = maxdiff_df.reindex(sorted(maxdiff_df.columns), axis=1)

        return maxdiff_df

    @staticmethod
    def format_group_metrics(
        groupmetrics_dfs: List[DataFrame],
        prob_values: List[str] = None,
        pred_values: str = None,
        datarole: str = "TEST",
    ) -> DataFrame:
        """
        Converts list of group metrics DataFrames to a single DataFrame

        Parameters
        ----------
        groupmetrics_dfs: list of pandas.DataFrame
            List of group metrics DataFrames generated by CASAction
        pred_values : str
            Required for regression problems, otherwise not used.
            Variable name containing the predicted values in score_table. The variable name must follow SAS naming
            conventions (no spaces and the name cannot begin with a number or symbol).Required for regression problems.
            The default value is None.
        prob_values : list of str
            Required for classification problems, otherwise not used
            A list of variable names containing the predicted probability values in the score table. The first element
            should represent the predicted probability of the target class. Required for classification problems. Default
            is None.
        datarole : str, optional
            The data being used to assess bias (i.e. 'TEST', 'VALIDATION', etc.). Default is 'TEST'.

        Returns
        -------
        pandas.DataFrame
            A singular DataFrame containing formatted data for group metrics
        """
        # adding group metrics dataframes and adding values/ formatting
        groupmetrics_df = pd.concat(groupmetrics_dfs)
        groupmetrics_df = groupmetrics_df.rename(
            columns={
                "Group": "LEVEL",
                "N": "nobs",
                "MISCEVENT": "misccutoff",
                "MISCEVENTKS": "miscks",
                "cutoffKS": "kscut",
                "PREDICTED": "avgyhat",
                "maxKS": "ks",
            }
        )
        groupmetrics_df["VLABEL"] = ""
        groupmetrics_df["_DATAROLE_"] = datarole

        for col in groupmetrics_df.columns:
            if prob_values is not None:
                upper_cols = [
                    "LEVEL",
                    "_VARIABLE_",
                    "_DATAROLE_",
                    "VLABEL",
                    "INTO_EVENT",
                    "PREDICTED_EVENT",
                ] + prob_values
            else:
                upper_cols = ["LEVEL", "_VARIABLE_", "_DATAROLE_", "VLABEL"] + [
                    pred_values
                ]
            if col not in upper_cols:
                groupmetrics_df = groupmetrics_df.rename(
                    columns={col: "_" + col.lower() + "_"}
                )

        groupmetrics_df = groupmetrics_df.reindex(
            sorted(groupmetrics_df.columns), axis=1
        )
        return groupmetrics_df

    @classmethod
    @experimental
    def bias_dataframes_to_json(
        cls,
        groupmetrics: DataFrame = None,
        maxdifference: DataFrame = None,
        n_sensitivevariables: int = None,
        actual_values: str = None,
        prob_values: List[str] = None,
        levels: List[str] = None,
        pred_values: str = None,
        json_path: Union[str, Path, None] = None,
    ):
        """
        Properly formats data from FairAITools CAS Action Set into a JSON readable formats

        Parameters
        ----------
        groupmetrics: pandas.DataFrame
            A DataFrame containing the group metrics data
        maxdifference: pandas.DataFrame
            A DataFrame containing the max difference data
        n_sensitivevariables: int
            The total number of sensitive values
        actual_values : str
            Variable name containing the actual values in score_table. The variable name must follow SAS naming
            conventions (no spaces and the name cannot begin with a number or symbol).
        prob_values : list of str
            Required for classification problems, otherwise not used
            A list of variable names containing the predicted probability values in the score table. The first element
            should represent the predicted probability of the target class. Required for classification problems. Default
            is None.
        levels: list of str
            Required for classification problems, otherwise not used
            List of classes of a nominal target in the order they were passed in `prob_values`. Levels must be passed as a
            string. Default is None.
        pred_values : str
            Required for regression problems, otherwise not used
            Variable name containing the predicted values in `score_table`. The variable name must follow SAS naming
            conventions (no spaces and the name cannot begin with a number or symbol). Required for regression problems.
            The default value is None.
        json_path : str or pathlib.Path, optional
            Location for the output JSON files. If a path is passed, the json files will populate in the directory and
            the function will return None, unless return_dataframes is True. Otherwise, the function will return the json
            strings in a dictionary (dict["maxDifferences.json"] and dict["groupMetrics.json"]). The default value is
            None.

        Returns
        -------
        dict
            Dictionary containing a key-value pair representing the files name and json
            dumps respectively.
        """
        folder = "reg_jsons" if prob_values is None else "clf_jsons"

        dfs = (maxdifference, groupmetrics)
        json_dict = [{}, {}]

        for i, name in enumerate(["maxDifferences", "groupMetrics"]):
            # reading template files
            json_template_path = (
                Path(__file__).resolve().parent / f"template_files/{folder}/{name}.json"
            )
            json_dict[i] = cls.read_json_file(json_template_path)
            # updating data rows
            for row_num in range(len(dfs[i])):
                row_dict = dfs[i].iloc[row_num].replace(float("nan"), None).to_dict()
                new_data = {"dataMap": row_dict, "rowNumber": row_num + 1}
                json_dict[i]["data"].append(new_data)

        # formatting metric label for max diff
        for i in range(n_sensitivevariables):
            if prob_values is not None:
                for j, prob_label in enumerate(prob_values):
                    json_dict[0]["data"][(i * 26) + j]["dataMap"][
                        "MetricLabel"
                    ] = f"Average Predicted: {actual_values}={levels[j]}"

            else:
                json_dict[0]["data"][i * 8]["dataMap"][
                    "MetricLabel"
                ] = f"Average Predicted: {actual_values}"

        # formatting parameter map for group metrics
        if prob_values is not None:
            for i, prob_label in enumerate(prob_values):
                paramdict = {
                    "label": prob_label,
                    "length": 8,
                    # TODO: figure out order ordering
                    "order": 34 + i,
                    "parameter": prob_label,
                    "preformatted": False,
                    "type": "num",
                    "values": [prob_label],
                }
                json_dict[1]["parameterMap"][prob_label] = paramdict
                # cls.add_dict_key(
                #     dict=json_dict[1]["parameterMap"],
                #     pos=i + 3,
                #     new_key=prob_label,
                #     new_value=paramdict,]
                # )

        else:
            json_dict[1]["parameterMap"]["predict"]["label"] = pred_values
            json_dict[1]["parameterMap"]["predict"]["parameter"] = pred_values
            json_dict[1]["parameterMap"]["predict"]["values"] = [pred_values]
            json_dict[1]["parameterMap"][pred_values] = json_dict[1]["parameterMap"][
                "predict"
            ]
            del json_dict[1]["parameterMap"]["predict"]

        if json_path:
            for i, name in enumerate([MAXDIFFERENCES, GROUPMETRICS]):
                with open(Path(json_path) / name, "w") as json_file:
                    json_file.write(json.dumps(json_dict[i], indent=4, cls=NpEncoder))
                if cls.notebook_output:
                    print(
                        f"{name} was successfully written and saved to "
                        f"{Path(json_path) / name}"
                    )
        else:
            return {
                MAXDIFFERENCES: json.dumps(json_dict[0], indent=4, cls=NpEncoder),
                GROUPMETRICS: json.dumps(json_dict[1], indent=4, cls=NpEncoder),
            }

    @classmethod
    def calculate_model_statistics(
        cls,
        target_value: Union[str, int, float],
        validate_data: Union[DataFrame, List[list], Type["numpy.ndarray"]] = None,
        train_data: Union[DataFrame, List[list], Type["numpy.ndarray"]] = None,
        test_data: Union[DataFrame, List[list], Type["numpy.ndarray"]] = None,
        json_path: Union[str, Path, None] = None,
        target_type: str = "classification",
        cutoff: Optional[float] = None,
    ) -> Union[dict, None]:
        """
        Calculates fit statistics (including ROC and Lift curves) from datasets and then
        either writes them to JSON files or returns them as a single dictionary.

        Calculations are performed using a call to SAS CAS via the swat package. An
        error will be raised if the swat package is not installed or if a connection to
        a SAS Viya system is not possible.

        Datasets must contain the actual and predicted values and may optionally contain
        the predicted probabilities. If no probabilities are provided, a dummy
        probability dataset is generated based on the predicted values and normalized by
        the target value.

        Datasets can be provided in the following forms, with the assumption that data
        is ordered as `actual`, `predict`, and `probability` respectively:

        * pandas dataframe: the actual and predicted values are their own columns
        * list: the actual and predicted values are their own indexed entry
        * numpy array: the actual and predicted values are their own columns or rows \
        and ordered such that the actual values come first and the predicted second

        If a json_path is supplied, then this function outputs a set of JSON files named
        "dmcas_fitstat.json", "dmcas_roc.json", "dmcas_lift.json".

        Parameters
        ----------
        target_value : str, int, or float
            Target event value for model prediction events.
        validate_data : pandas.DataFrame, list of list, or numpy.ndarray, optional
            Dataset pertaining to the validation data. The default value is None.
        train_data : pandas.DataFrame, list of list, or numpy.ndarray, optional
            Dataset pertaining to the training data. The default value is None.
        test_data : pandas.DataFrame, list of list, or numpy.ndarray, optional
            Dataset pertaining to the test data. The default value is None.
        json_path : str or pathlib.Path, optional
            Location for the output JSON files. The default value is None.
        target_type: str, optional
            Type of target the model is trying to find. Currently supports "classification"
            and "prediction" types. The default value is "classification".

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

            data = cls.stat_dataset_to_dataframe(data, target_value, target_type)
            data["predict_proba2"] = 1 - data["predict_proba"]

            conn.upload(
                data,
                casout={"caslib": "Public", "name": "assess_dataset", "replace": True},
            )

            if target_type == "classification":
                conn.percentile.assess(
                    table={"name": "assess_dataset", "caslib": "Public"},
                    inputs="predict_proba",
                    response="actual",
                    event="1",
                    pvar="predict_proba2",
                    pevent="0",
                    includeLift=True,
                    fitStatOut={"name": "FitStat", "replace": True, "caslib": "Public"},
                    rocOut={"name": "ROC", "replace": True, "caslib": "Public"},
                    casout={"name": "Lift", "replace": True, "caslib": "Public"},
                )
            else:
                conn.percentile.assess(
                    table={"name": "assess_dataset", "caslib": "Public"},
                    response="actual",
                    inputs="predict",
                    fitStatOut={"caslib": "Public", "name": "FitStat", "replace": True},
                    casout={"caslib": "Public", "name": "Lift", "replace": True},
                )

            fitstat_dict = (
                pd.DataFrame(conn.CASTable("FitStat", caslib="Public").to_frame())
                .transpose()
                .squeeze()
                .to_dict()
            )
            json_dict[0]["data"][i]["dataMap"].update(fitstat_dict)

            if target_type == "classification":
                roc_df = pd.DataFrame(conn.CASTable("ROC", caslib="Public").to_frame())
                roc_dict = cls.apply_dataframe_to_json(json_dict[1]["data"], i, roc_df)
                for j in range(len(roc_dict)):
                    json_dict[1]["data"][j].update(roc_dict[j])
                    fitstat_data = None
                    if roc_dict[j]["dataMap"]["_KS_"] == 1:
                        fitstat_data = dict()
                        missing_stats = (
                            "_KS_",
                            "_KS2_",
                            "_C_",
                            "_Gini_",
                            "_Gamma_",
                            "_Tau_",
                        )
                        for stat in missing_stats:
                            if stat in roc_dict[j]["dataMap"]:
                                fitstat_data[stat] = roc_dict[j]["dataMap"][stat]
                    if fitstat_data:
                        json_dict[0]["data"][i]["dataMap"].update(fitstat_data)

            lift_df = pd.DataFrame(conn.CASTable("Lift", caslib="Public").to_frame())
            lift_dict = cls.apply_dataframe_to_json(json_dict[2]["data"], i, lift_df, 1)
            for j in range(len(lift_dict)):
                json_dict[2]["data"][j].update(lift_dict[j])

        if json_path:
            for i, name in enumerate([FITSTAT, ROC, LIFT]):
                if not (name == ROC and target_type == "prediction"):
                    with open(Path(json_path) / name, "w") as json_file:
                        json_file.write(
                            json.dumps(json_dict[i], indent=4, cls=NpEncoder)
                        )
                    if cls.notebook_output:
                        print(
                            f"{name} was successfully written and saved to "
                            f"{Path(json_path) / name}"
                        )
        else:
            if target_type == "classification":
                return {
                    FITSTAT: json.dumps(json_dict[0], indent=4, cls=NpEncoder),
                    ROC: json.dumps(json_dict[1], indent=4, cls=NpEncoder),
                    LIFT: json.dumps(json_dict[2], indent=4, cls=NpEncoder),
                }
            else:
                return {
                    FITSTAT: json.dumps(json_dict[0], indent=4, cls=NpEncoder),
                    LIFT: json.dumps(json_dict[2], indent=4, cls=NpEncoder),
                }

    @staticmethod
    def check_for_data(
        validate: Union[DataFrame, List[list], Type["numpy.ndarray"]] = None,
        train: Union[DataFrame, List[list], Type["numpy.ndarray"]] = None,
        test: Union[DataFrame, List[list], Type["numpy.ndarray"]] = None,
    ) -> list:
        """
        Check which datasets were provided and return a list of flags.

        Parameters
        ----------
        validate : pandas.DataFrame, list of list, or numpy.ndarray, optional
            Dataset pertaining to the validation data. The default value is None.
        train : pandas.DataFrame, list of list, or numpy.ndarray, optional
            Dataset pertaining to the training data. The default value is None.
        test : pandas.DataFrame, list of list, or numpy.ndarray, optional
            Dataset pertaining to the test data. The default value is None.

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
    def stat_dataset_to_dataframe(
        data: Union[DataFrame, List[list], Type["numpy.ndarray"]],
        target_value: Union[str, int, float] = None,
        target_type: str = "classification",
    ) -> DataFrame:
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
        data : pandas.DataFrame, list of list, or numpy.ndarray
            Dataset representing the actual and predicted values of the model. May also
            include the prediction probabilities.
        target_value : str, int, or float, optional
            Target event value for model prediction events. Used for creating a binary
            probability column when no probability values are provided. The default
            value is None.

        Returns
        -------
        data : pandas.DataFrame
            Dataset formatted for SAS CAS upload.

        Raises
        ------
        ValueError
            Raised if an improper data format is provided.

        """
        # Convert target_value to numeric for creating binary probabilities
        if isinstance(target_value, str):
            target_value = float(target_value)

        # Assume column order (actual, predicted, probability) per argument instructions
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 2:
                data.columns = ["actual", "predict"]
                if target_type == "classification":
                    data["predict_proba"] = data["predict"].gt(target_value).astype(int)
            elif len(data.columns) == 3:
                data.columns = ["actual", "predict", "predict_proba"]
        elif isinstance(data, list):
            if len(data) == 2:
                data = pd.DataFrame({"actual": data[0], "predict": data[1]})
                if target_type == "classification":
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
                if target_type == "classification":
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
    def apply_dataframe_to_json(
        json_dict: dict, partition: int, stat_df: DataFrame, is_lift: bool = False
    ) -> dict:
        """
        Map the values of the ROC or Lift charts from SAS CAS to the dictionary
        representation of the respective json file.

        Parameters
        ----------
        json_dict : dict
            Dictionary representation of the ROC or Lift chart json file.
        partition : int
            Numerical representation of the data partition. Either 0, 1, or 2.
        stat_df : pandas.DataFrame
            ROC or Lift DataFrame generated from the SAS CAS percentile action set.
        is_lift : bool
            Specify whether to use logic for Lift or ROC row counting. Default value is
            False.

        Returns
        -------
        json_dict : dict
            Dictionary representation of the ROC or Lift chart json file, with the
            values from the SAS CAS percentile action set added in.
        """
        for row_num in range(len(stat_df)):
            row_dict = stat_df.iloc[row_num].replace(float("nan"), None).to_dict()
            if is_lift:
                json_dict[(row_num + partition + 1) + partition * len(stat_df)][
                    "dataMap"
                ].update(row_dict)
            else:
                json_dict[row_num + (partition * len(stat_df))]["dataMap"].update(
                    row_dict
                )
        return json_dict

    @staticmethod
    def read_json_file(path: Union[str, Path]) -> Any:
        """
        Reads a JSON file from a given path.

        Parameters
        ----------
        path : str or pathlib.Path
            Location of the JSON file to be opened.

        Returns
        -------
        json.load(jFile) : str
            String contents of JSON file.
        """
        with open(path) as jFile:
            return json.load(jFile)

    @staticmethod
    def format_parameter(param_name: str):
        """
        Formats the parameter name to the JSON standard expected for dmcas_fitstat.json.

        Parameters
        ----------
        param_name : str
            Name of the parameter.

        Returns
        -------
        str
            Name of the parameter.
        """
        if not (param_name.startswith("_") and param_name.endswith("_")):
            if not param_name.startswith("_"):
                param_name = "_" + param_name
            if not param_name.endswith("_"):
                param_name = param_name + "_"

        return param_name

    @staticmethod
    def convert_data_role(data_role: Union[str, int]) -> Union[str, int]:
        """
        Converts the data role identifier from string to int or int to string.

        JSON file descriptors require the string, int, and formatted int. If the
        provided data role is not valid, defaults to TRAIN (1).

        Parameters
        ----------
        data_role : str or int
            Identifier of the data set's role; either TRAIN, TEST, or VALIDATE, or
            correspondingly 1, 2, or 3.

        Returns
        -------
        conversion : str or int
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
    def create_requirements_json(
        cls,
        model_path: Union[str, Path, None] = Path.cwd(),
        output_path: Union[str, Path, None] = None,
    ) -> Union[dict, None]:
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
        model_path : str or pathlib.Path, optional
            The path to a Python project, by default the current working directory.
        output_path : str or pathlib.Path, optional
            The path for the output requirements.json file. The default value is None.

        Returns
        -------
        list of dict
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
            return json_dicts

    @staticmethod
    def get_local_package_version(package_list: List[str]) -> List[List[str]]:
        """
        Get package_name versions from the local environment.

        If the package_name does not contain an attribute of "__version__",
        "version", or "VERSION", no package_name version will be found.

        Parameters
        ----------
        package_list : list of str
            List of Python packages.

        Returns
        -------
        list of list of str
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
    def get_code_dependencies(
        cls, model_path: Union[str, Path] = Path.cwd()
    ) -> List[str]:
        """
        Get the package dependencies for all Python scripts in the provided directory
        path.

        Note that currently this functionality only works for .py files.

        Parameters
        ----------
        model_path : str or pathlib.Path, optional
            File location for the output JSON file. The default value is the current
            working directory.

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
    def find_imports(file_path: Union[str, Path]) -> List[str]:
        """
        Find import calls in provided Python code path.

        Ignores built in Python modules.

        Credit: modified from https://stackoverflow.com/questions/44988487/regex-to
        -parse-import-statements-in-python

        Parameters
        ----------
        file_path : str or pathlib.Path
            File location for the Python file to be parsed.

        Returns
        -------
        list of str
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
    def get_pickle_file(pickle_folder: Union[str, Path] = Path.cwd()) -> List[Path]:
        """
        Given a file path, retrieve the pickle file(s).

        Parameters
        ----------
        pickle_folder : str or pathlib.Path
            File location for the input pickle file. The default value is the current
            working directory.

        Returns
        -------
        list of pathlib.Path
            A list of pickle files.
        """
        return [
            p for p in Path(pickle_folder).iterdir() if p.suffix in [".pickle", ".pkl"]
        ]

    @classmethod
    def get_pickle_dependencies(cls, pickle_file: Union[str, Path]) -> List[str]:
        """
        Reads the pickled byte stream from a file object, serializes the pickled byte
        stream as a bytes object, and inspects the bytes object for all Python
        modules and aggregates them in a list.

        Parameters
        ----------
        pickle_file : str or pathlib.Path
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
    def get_package_names(stream: Union[bytes, str]) -> List[str]:
        """
        Generates a list of found `package` names from a pickle stream.

        In most cases, the `packages` returned by the function will be valid Python
        packages. A check is made in get_local_package_version to ensure that the
        package is in fact a valid Python package.

        This code has been adapted from the following stackoverflow example and
        utilizes the pickletools package.

        Credit: modified from
        https://stackoverflow.com/questions/64850179/inspecting-a-pickle-dump-for-dependencies

        More information here:
        https://github.com/python/cpython/blob/main/Lib/pickletools.py

        Parameters
        ----------
        stream : bytes or str
            A file like object or string containing the pickle.

        Returns
        -------
        list of str
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
    def remove_standard_library_packages(package_list: List[str]) -> List[str]:
        """
        Remove any packages from the required list of installed packages that are
        part of the Python Standard Library.

        Parameters
        ----------
        package_list : list of str
            List of all packages found that are not Python built-in packages.

        Returns
        -------
        list of str
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

    @classmethod
    def generate_model_card(
        cls,
        model_prefix: str,
        model_files: Union[str, Path, dict],
        algorithm: str,
        train_data: pd.DataFrame,
        train_predictions: Union[pd.Series, list],
        target_type: str = "classificaiton",
        target_value: Union[str, int, float, None] = None,
        interval_vars: Optional[list] = [],
        class_vars: Optional[list] = [],
        selection_statistic: str = None,
        training_table_name: str = None,
        server: str = "cas-shared-default",
        caslib: str = "Public",
    ):
        """
        Generates everything required for the model card feature within SAS Model Manager.

        This includes uploading the training data to CAS, updating ModelProperties.json to have
        some extra properties, and generating dmcas_relativeimportance.json.

        Parameters
        ----------
        model_prefix : string
            The prefix used to name files relating to the model. This is used to provide a unique
            name to the training data table when it is uploaded to CAS.
        model_files : string, Path, or dict
            Either the directory location of the model files (string or Path object), or
            a dictionary containing the contents of all the model files.
        algorithm : str
            The name of the algorithm used to generate the model.
        train_data: pandas.DataFrame
            Training data that contains all input variables as well as the target variable.
        train_predictions : pandas.Series, list
            List of predictions made by the model on the training data.
        target_type : string
            Type of target the model is trying to find. Currently supports "classification" and "prediction" types.
            The default value is "classification".
        target_value : string, int, float, optional
            Value the model is targeting for classification models. This argument is not needed for
            prediction models. The default value is None.
        interval_vars : list, optional
            A list of interval variables. The default value is an empty list.
        class_vars : list, optional
            A list of classification variables. The default value is an empty list.
        selection_statistic: str, optional
            The selection statistic chosen to score the model against other models. Classification
            models can take any of the following values: "_RASE_", "_GINI_", "_GAMMA_", "_MCE_",
            "_ASE_", "_MCLL_", "_KS_", "_KSPostCutoff_", "_DIV_", "_TAU_", "_KSCut_", or "_C_".
            Prediction models can take any of the following values: "_ASE_", "_DIV_", "_RASE_", "_MAE_",
            "_RMAE_", "_MSLE_", "_RMSLE_" The default value is "_KS_" for classification models and
            "_ASE_" for prediction models.
        server: str, optional
            The CAS server the training data will be stored on. The default value is "cas-shared-default"
        caslib: str, optional
            The caslib the training data will be stored on. The default value is "Public"
        """
        if not target_value and target_type == "classification":
            raise RuntimeError(
                "For the model card data to be properly generated on a classification "
                "model, a target value is required."
            )
        if target_type not in ["classification", "prediction"]:
            raise RuntimeError(
                "Only classification and prediction target types are currently accepted."
            )
        if selection_statistic is None:
            if target_type == "classification":
                selection_statistic = "_KS_"
            elif target_type == "prediction":
                selection_statistic = "_ASE_"
        if selection_statistic not in cls.valid_params:
            raise RuntimeError(
                "The selection statistic must be a value generated in dmcas_fitstat.json. See "
                "the documentation for a list of valid selection statistic values."
            )
        if not algorithm:
            raise RuntimeError(
                "Either a given algorithm or a model is required for the model card."
            )
        try:
            sess = current_session()
            conn = sess.as_swat()
        except ImportError:
            raise RuntimeError(
                "The `swat` package is required to generate fit statistics, ROC, and "
                "Lift charts with the calculate_model_statistics function."
            )

        # Upload training table to CAS. The location of the training table is returned.
        training_table = cls.upload_training_data(
            conn, model_prefix, train_data, training_table_name, server, caslib
        )

        # Generates the event percentage for Classification targets, and the event average
        # for prediction targets
        update_dict = cls.generate_outcome_average(
            train_data=train_data,
            input_variables=interval_vars + class_vars,
            target_type=target_type,
            target_value=target_value,
        )

        # Formats all new ModelProperties information into one dictionary that can be used to update the json file
        update_dict["trainTable"] = training_table
        update_dict["selectionStatistic"] = selection_statistic
        update_dict["algorithm"] = algorithm
        update_dict["selectionStatisticValue"] = cls.get_selection_statistic_value(
            model_files, selection_statistic
        )
        cls.update_model_properties(model_files, update_dict)

        # Generates dmcas_relativeimportance.json file
        cls.generate_variable_importance(
            conn,
            model_files,
            train_data,
            train_predictions,
            target_type,
            interval_vars,
            class_vars,
            caslib,
        )

        # Generates dmcas_misc.json file
        if target_type == "classification":
            cls.generate_misc(model_files)

    @staticmethod
    def upload_training_data(
        conn,
        model_prefix: str,
        train_data: pd.DataFrame,
        train_data_name: str,
        server: str = "cas-shared-default",
        caslib: str = "Public",
    ):
        """
        Uploads training data to CAS server.

        Parameters
        ----------
        conn
            SWAT connection. Used to connect to CAS server.
        model_prefix : string
            The prefix used to name files relating to the model. This is used to provide a unique
            name to the training data table when it is uploaded to CAS.
        train_data: pandas.DataFrame
            Training data that contains all input variables as well as the target variable.
        server: str, optional
            The CAS server the training data will be stored on. The default value is "cas-shared-default"
        caslib: str, optional
            The caslib the training data will be stored on. The default value is "Public"

        Returns
        -------
        string
        Returns a string that represents the location of the training table within CAS.
        """
        # Upload raw training data to caslib so that data can be analyzed
        if not train_data_name:
            train_data_name = model_prefix + "_train_data"
        upload_train_data = conn.upload(
            train_data, casout={"name": train_data_name, "caslib": caslib}, promote=True
        )

        if upload_train_data.status is not None:
            # raise RuntimeError(
            warnings.warn(
                f"A table with the name {train_data_name} already exists in the specified caslib. If this "
                f"is not intentional, please either rename the training data file or remove the duplicate from "
                f"the caslib."
            )

        return server + "/" + caslib + "/" + train_data_name.upper()

    @staticmethod
    def generate_outcome_average(
        train_data: pd.DataFrame,
        input_variables: list,
        target_type,
        target_value: Union[str, int, float] = None,
    ):
        """
        Generates the outcome average of the training data. For prediction targets, the event average
        is generated. For Classification targets, the event percentage is returned.

        Parameters
        ----------
        train_data: pandas.DataFrame
            Training data that contains all input variables as well as the target variable. If multiple
            non-input variables are included, the function will assume that the first non-input variable row
            is the output.
        input_variables: list
            A list of all input variables used by the model. Used to isolate the output variable.
        target_type : string
            Type the model is targeting. Currently supports "classification" and "prediction" types.
        target_value : string, int, float, optional
            Value the model is targeting for Classification models. This argument is not needed for
            prediction models. The default value is None.

        Returns
        -------
        dict
        Returns a dictionary with a key value pair that represents the outcome average.
        """
        import numbers

        output_var = train_data.drop(input_variables, axis=1)
        if target_type == "classification":
            value_counts = output_var[output_var.columns[0]].value_counts()
            return {"eventPercentage": value_counts[target_value] / sum(value_counts)}
        elif target_type == "prediction":
            if not isinstance(
                output_var[output_var.columns[0]].iloc[0], numbers.Number
            ):
                raise ValueError(
                    "Detected output column is not numeric. Please ensure that "
                    + "the correct output column is being passed, and that no extra columns "
                    + "are in front of the output column. This function assumes that the first "
                    + "non-input column is the output column.jf"
                )
            return {
                "eventAverage": sum(output_var[output_var.columns[0]]) / len(output_var)
            }

    @staticmethod
    def get_selection_statistic_value(
        model_files: Union[str, Path, dict], selection_statistic: str = "_GINI_"
    ):
        """
        Finds the value of the chosen selection statistic in dmcas_fitstat.json, which should have been
        generated before this function has been called.

        Parameters
        ----------
        model_files : string, Path, or dict
            Either the directory location of the model files (string or Path object), or
            a dictionary containing the contents of all the model files.
        selection_statistic: str, optional
            The selection statistic chosen to score the model against other models. Can be any of the
            following values: "_RASE_", "_NObs_", "_GINI_", "_GAMMA_", "_MCE_", "_ASE_", "_MCLL_",
            "_KS_", "_KSPostCutoff_", "_DIV_", "_TAU_", "_KSCut_", or "_C_". The default value is "_GINI_".

        Returns
        -------
        float
        Returns the numerical value assoicated with the chosen selection statistic.
        """
        if isinstance(model_files, dict):
            if FITSTAT not in model_files:
                raise RuntimeError(
                    "The dmcas_fitstat.json file must be generated before the model card data "
                    "can be generated."
                )
            for fitstat in model_files[FITSTAT]["data"]:
                if fitstat["dataMap"]["_DataRole_"] == "TRAIN":
                    if (
                        selection_statistic not in fitstat["dataMap"]
                        or fitstat["dataMap"][selection_statistic] == None
                    ):
                        raise RuntimeError(
                            "The chosen selection statistic was not generated properly. Please ensure the value has been "
                            "properly created then try again."
                        )
                    return fitstat["dataMap"][selection_statistic]
        else:
            if not Path.exists(Path(model_files) / FITSTAT):
                raise RuntimeError(
                    "The dmcas_fitstat.json file must be generated before the model card data "
                    "can be generated."
                )
            with open(Path(model_files) / FITSTAT, "r") as fitstat_json:
                fitstat_dict = json.load(fitstat_json)
                for fitstat in fitstat_dict["data"]:
                    if fitstat["dataMap"]["_DataRole_"] == "TRAIN":
                        if (
                            selection_statistic not in fitstat["dataMap"]
                            or fitstat["dataMap"][selection_statistic] == None
                        ):
                            raise RuntimeError(
                                "The chosen selection statistic was not generated properly. Please ensure the value has been "
                                "properly created then try again."
                            )
                        return fitstat["dataMap"][selection_statistic]

    @staticmethod
    def update_model_properties(model_files, update_dict):
        """
        Updates the ModelProperties.json file to include properties listed in the update_dict dictionary.

        Parameters
        ----------
        model_files : string, Path, or dict
            Either the directory location of the model files (string or Path object), or
            a dictionary containing the contents of all the model files.
        update_dict : dictionary
            A dictionary containing the key-value pairs that represent properties to be added
            to the ModelProperties.json file.
        """
        if isinstance(model_files, dict):
            if PROP not in model_files:
                raise RuntimeError(
                    "The ModelProperties.json file must be generated before the model card data "
                    "can be generated."
                )
            for key in update_dict:
                if not isinstance(update_dict[key], str):
                    model_files[PROP][key] = str(round(update_dict[key], 14))
                else:
                    model_files[PROP][key] = update_dict[key]
        else:
            if not Path.exists(Path(model_files) / PROP):
                raise RuntimeError(
                    "The ModelProperties.json file must be generated before the model card data "
                    "can be generated."
                )
            with open(Path(model_files) / PROP, "r+") as properties_json:
                model_properties = json.load(properties_json)
                for key in update_dict:
                    if not isinstance(update_dict[key], str):
                        model_properties[key] = str(round(update_dict[key], 14))
                    else:
                        model_properties[key] = update_dict[key]
                properties_json.seek(0)
                properties_json.write(
                    json.dumps(model_properties, indent=4, cls=NpEncoder)
                )
                properties_json.truncate()

    @classmethod
    def generate_variable_importance(
        cls,
        conn,
        model_files: Union[str, Path, dict],
        train_data: pd.DataFrame,
        train_predictions: Union[pd.Series, list],
        target_type: str = "classification",
        interval_vars: Optional[list] = [],
        class_vars: Optional[list] = [],
        caslib: str = "Public",
    ):
        """
        Generates the dmcas_relativeimportance.json file, which is used to determine variable importance

        Parameters
        ----------
        conn
            A SWAT connection used to connect to the user's CAS server
        model_files : string, Path, or dict
            Either the directory location of the model files (string or Path object), or
            a dictionary containing the contents of all the model files.
        train_data: pandas.DataFrame
            Training data that contains all input variables as well as the target variable.
        train_predictions : pandas.Series, list
            List of predictions made by the model on the training data.
        target_type : string, optional
            Type the model is targeting. Currently supports "classification" and "prediction" types.
            The default value is "classification".
        interval_vars : list, optional
            A list of interval variables. The default value is an empty list.
        class_vars : list, optional
            A list of classification variables. The default value is an empty list.
        caslib: str, optional
            The caslib the training data will be stored on. The default value is "Public"
        """
        # Remove target variable from training data by selecting only input variable columns
        x_train_data = train_data[interval_vars + class_vars]
        # Upload scored training data to run variable importance on
        x_train_data.insert(0, "Prediction", train_predictions, True)
        conn.upload(
            x_train_data,
            casout={"name": "train_data", "replace": True, "caslib": caslib},
        )

        # Load actionset necessary to generate variable importance
        conn.loadactionset("dataPreprocess")
        request_packages = list()
        if target_type == "classification":
            method = "DTREE"
            treeCrit = "Entropy"
        elif target_type == "prediction":
            method = "RTREE"
            treeCrit = "RSS"
        else:
            raise RuntimeError(
                "The selected model type is unsupported. Currently, only models that have prediction or classification target types are supported."
            )
        request_packages = list()
        if interval_vars:
            request_packages.append(
                {
                    "name": "BIN",
                    "inputs": [{"name": var} for var in interval_vars],
                    "targets": [{"name": "Prediction"}],
                    "discretize": {
                        "method": method,
                        "arguments": {
                            "minNBins": 1,
                            "maxNBins": 8,
                            "treeCrit": treeCrit,
                            "contingencyTblOpts": {
                                "inputsMethod": "BUCKET",
                                "inputsNLevels": 100,
                            },
                            "overrides": {
                                "minNObsInBin": 5,
                                "binMissing": True,
                                "noDataLowerUpperBound": True,
                            },
                        },
                    },
                }
            )
        if class_vars:
            request_packages.append(
                {
                    "name": "BIN_NOM",
                    "inputs": [{"name": var} for var in class_vars],
                    "targets": [{"name": "Prediction"}],
                    "catTrans": {
                        "method": method,
                        "arguments": {
                            "minNBins": 1,
                            "maxNBins": 8,
                            "treeCrit": treeCrit,
                            "overrides": {"minNObsInBin": 5, "binMissing": True},
                        },
                    },
                }
            )
        var_data = conn.dataPreprocess.transform(
            table={"name": "train_data", "caslib": caslib},
            requestPackages=request_packages,
            evaluationStats=True,
            percentileMaxIterations=10,
            percentileTolerance=0.00001,
            distinctCountLimit=5000,
            sasVarNameLength=True,
            outputTableOptions={"inputVarPrintOrder": True},
            sasProcClient=True,
        )
        var_importances = var_data["VarTransInfo"][["Variable", "RelVarImportance"]]
        var_importances = var_importances.sort_values(
            by=["RelVarImportance"], ascending=False
        ).reset_index(drop=True)
        relative_importances = list()
        for index, row in var_importances.iterrows():
            if row["Variable"] in interval_vars:
                level = "INTERVAL"
            elif row["Variable"] in class_vars:
                level = "NOMINAL"
            relative_importances.append(
                {
                    "dataMap": {
                        "LABEL": "",
                        "LEVEL": level,
                        "ROLE": "INPUT",
                        "RelativeImportance": str(row["RelVarImportance"]),
                        "Variable": row["Variable"],
                    },
                    "rowNumber": index + 1,
                }
            )
        json_template_path = (
            Path(__file__).resolve().parent / f"template_files/{VARIMPORTANCES}"
        )
        with open(json_template_path, "r") as f:
            relative_importance_json = json.load(f)
        relative_importance_json["data"] = relative_importances

        if isinstance(model_files, dict):
            model_files[VARIMPORTANCES] = json.dumps(
                relative_importance_json, indent=4, cls=NpEncoder
            )
            if cls.notebook_output:
                print(
                    f"{VARIMPORTANCES} was successfully written and saved to "
                    f"model files dictionary."
                )
        else:
            with open(Path(model_files) / VARIMPORTANCES, "w") as json_file:
                json_file.write(
                    json.dumps(relative_importance_json, indent=4, cls=NpEncoder)
                )
            if cls.notebook_output:
                print(
                    f"{VARIMPORTANCES} was successfully written and saved to "
                    f"{Path(model_files) / VARIMPORTANCES}"
                )

    @classmethod
    def generate_misc(cls, model_files: Union[str, Path, dict]):
        """
        Generates the dmcas_misc.json file, which is used to determine variable importance

        Parameters
        ----------
        conn
            A SWAT connection used to connect to the user's CAS server
        model_files : string, Path, or dict
            Either the directory location of the model files (string or Path object), or
            a dictionary containing the contents of all the model files.
        """
        if isinstance(model_files, dict):
            if ROC not in model_files:
                raise RuntimeError(
                    "The dmcas_roc.json file must be generated before the model card data "
                    "can be generated."
                )
            roc_table = model_files[ROC]
        else:
            if not Path.exists(Path(model_files) / ROC):
                raise RuntimeError(
                    "The dmcas_roc.json file must be generated before the model card data "
                    "can be generated."
                )
            with open(Path(model_files) / ROC, "r") as roc_file:
                roc_table = json.load(roc_file)
        correct_text = ["CORRECT", "INCORRECT", "CORRECT", "INCORRECT"]
        outcome_values = ["1", "0", "0", "1"]
        target_texts = ["Event", "Event", "NEvent", "NEvent"]
        target_values = ["1", "1", "0", "0"]

        misc_data = list()
        # Iterates through ROC table to get TRAIN, TEST, and VALIDATE data with a cutoff of .5
        for i in range(50, 300, 100):
            roc_data = roc_table["data"][i]["dataMap"]
            correctness_values = [
                roc_data["_TP_"],
                roc_data["_FP_"],
                roc_data["_TN_"],
                roc_data["_FN_"],
            ]
            for c_text, c_val, o_val, t_txt, t_val in zip(
                correct_text,
                correctness_values,
                outcome_values,
                target_texts,
                target_values,
            ):
                misc_data.append(
                    {
                        "dataMap": {
                            "CorrectText": c_text,
                            "Outcome": o_val,
                            "_Count_": f"{c_val}",
                            "_DataRole_": roc_data["_DataRole_"],
                            "_cutoffSource_": "Default",
                            "_cutoff_": "0.5",
                            "TargetText": t_txt,
                            "Target": t_val,
                        },
                        "rowNumber": len(misc_data) + 1,
                    }
                )

        json_template_path = Path(__file__).resolve().parent / f"template_files/{MISC}"
        with open(json_template_path, "r") as f:
            misc_json = json.load(f)
        misc_json["data"] = misc_data

        if isinstance(model_files, dict):
            model_files[MISC] = json.dumps(misc_json, indent=4, cls=NpEncoder)
            if cls.notebook_output:
                print(
                    f"{MISC} was successfully written and saved to "
                    f"model files dictionary."
                )
        else:
            with open(Path(model_files) / MISC, "w") as json_file:
                json_file.write(json.dumps(misc_json, indent=4, cls=NpEncoder))
            if cls.notebook_output:
                print(
                    f"{MISC} was successfully written and saved to "
                    f"{Path(model_files) / MISC}"
                )
