# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
from uuid import UUID
from warnings import warn

from pandas import DataFrame
import json

from .._services.model_repository import ModelRepository as mr
from ..core import PagedList, RestObj, current_session
from ..utils.misc import check_if_jupyter
from .write_score_code import ScoreCode as sc
from .zip_model import ZipModel as zm
from ..tasks import _create_project, _update_properties, _compare_properties


def get_model_properties(
    target_values: Union[list, str, None] = None,
    model_files: Union[str, Path, None] = None,
):
    if type(model_files) is dict:
        try:
            model = json.loads(model_files["ModelProperties.json"])
        except (json.JSONDecodeError, TypeError):
            model = model_files["ModelProperties.json"]

        try:
            input_var = json.loads(model_files["inputVar.json"])
        except (json.JSONDecodeError, TypeError):
            input_var = model_files["inputVar.json"]

        try:
            output_var = json.loads(model_files["outputVar.json"])
        except (json.JSONDecodeError, TypeError):
            output_var = model_files["outputVar.json"]

    else:
        with open(Path(model_files) / "ModelProperties.json") as f:
            model = json.load(f)
        with open(Path(model_files) / "inputVar.json") as f:
            input_var = json.load(f)
        with open(Path(model_files) / "outputVar.json") as f:
            output_var = json.load(f)
    if target_values is not None:
        target_string = ", ".join(target_values)
        model["classTargetValues"] = target_string
    for var in output_var:
        var["role"] = "output"
    for var in input_var:
        var["role"] = "input"
    return model, input_var, output_var


def project_exists(
    project: Union[str, dict, RestObj],
    response: Union[str, dict, RestObj, None] = None,
    target_values: Union[list, str, None] = None,
    model_files: Union[str, Path, None] = None,
    overwrite_project_properties: Optional[bool] = False,
) -> RestObj:
    """
    Checks if project exists on SAS Viya. If the project does not exist, then a new
    project is created or an error is raised.

    Parameters
    ----------
    project : str, dict, or RestObj
        The name or id of the model project, or a dictionary representation of the
        project.
    response : str, dict, or RestObj, optional
        JSON response of the get_project() call to model repository service.
    target_values : list of str, optional
        A list of target values for the target variable. This argument and the
        score_metrics argument dictate the handling of the predicted values from
        the prediction method. The default value is None.
    model_files : str, pathlib.Path, or dict
        Either the directory location of the model files (string or Path object), or
        a dictionary containing the contents of all the model files.
    overwrite_project_properties : bool, optional
        Set whether the project properties should be overwritten when attempting to
        import the model. The default value is False.

    Returns
    -------
    response : RestObj
        JSON response of the get_project() call to model repository service.

    Raises
    ------
    SystemError
        Alerts user that API calls cannot continue until a valid project is provided.
    """
    if response is None:
        try:
            warn(f"No project with the name or UUID {project} was found.")
            UUID(project)
            raise SystemError(
                "The provided UUID does not match any projects found in SAS Model "
                "Manager. Please enter a valid UUID or a new name for a project to be "
                "created."
            )
        except ValueError:
            repo = mr.default_repository().get("id")
            if model_files is not None:
                model, input_var, output_var = get_model_properties(
                    target_values, model_files
                )
                response = _create_project(project, model, repo, input_var, output_var)
            else:
                response = mr.create_project(project, repo)

            if check_if_jupyter():
                print(f"A new project named {response.name} was created.")
            return response
    else:
        model, input_var, output_var = get_model_properties(target_values, model_files)
        if overwrite_project_properties:
            response = _update_properties(project, model, input_var, output_var)
        else:
            _compare_properties(project, model, input_var, output_var)
        return response


def model_exists(
    project: Union[str, dict, RestObj],
    name: str,
    force: bool = "False",
    version_name: str = "latest",
) -> None:
    """
    Checks if model already exists in the same project and either raises an error or
    delete the redundant model. If no project version is provided, the version is
    assumed to be "latest".

    Parameters
    ----------
    project : str, dict, or RestObj
        The name or id of the model project, or a dictionary representation of the
        project.
    name : str
        The name of the model.
    force : bool, optional
        Sets whether to overwrite models with the same name upon upload. The default
        value is False.
    version_name : str, optional
        Name of project version to check if a model of the same name already exists.
        The default value is "latest".

    Raises
    ------
    ValueError
        Model repository API cannot overwrite an already existing model with the upload
        model call. Alerts user of the force argument to allow model overwriting.
    """
    project = mr.get_project(project)
    project_id = project["id"]
    project_versions = mr.list_project_versions(project)
    if version_name == "latest":
        version_name = project["latestVersion"]
    for version in project_versions:
        if version_name == version["name"]:
            version_id = version["id"]
            break
    project_models = mr.get(
        f"/projects/{project_id}/projectVersions/{version_id}/models"
    )

    if not project_models:
        return
    elif isinstance(project_models, RestObj):
        if project_models["name"] == name and force:
            mr.delete_model(project_models.id)
        elif project_models["name"] == name and not force:
            raise ValueError(
                f"A model with the same model name exists in project "
                f"{project.name}. Include the force=True argument to overwrite "
                f"models with the same name."
            )
    elif isinstance(project_models, PagedList):
        for model in project_models:
            if model["name"] == name and force:
                mr.delete_model(model.id)
            elif model["name"] == name and not force:
                raise ValueError(
                    f"A model with the same model name exists in project "
                    f"{project.name}. Include the force=True argument to overwrite "
                    f"models with the same name."
                )


class ImportModel:
    notebook_output = check_if_jupyter()

    @classmethod
    def import_model(
        cls,
        model_files: Union[str, Path, dict],
        model_prefix: str,
        project: Union[str, dict, RestObj],
        input_data: Optional[DataFrame] = None,
        predict_method: Union[Callable[..., List], List[Any]] = None,
        score_metrics: Optional[List[str]] = None,
        pickle_type: str = "pickle",
        project_version: str = "latest",
        missing_values: Optional[bool] = False,
        overwrite_model: Optional[bool] = False,
        score_cas: Optional[bool] = True,
        mlflow_details: Optional[dict] = None,
        predict_threshold: Optional[float] = None,
        target_values: Optional[List[str]] = None,
        overwrite_project_properties: Optional[bool] = False,
        target_index: Optional[int] = None,
        preprocess_function: Optional[Callable[[DataFrame], DataFrame]] = None,
        **kwargs,
    ) -> Tuple[RestObj, Union[dict, str, Path]]:
        """
        Import a model to SAS Model Manager using pzmm submodule.

        Using pzmm, generate Python score code and import the model files into
        SAS Model Manager. This function automatically checks the version of SAS
        Viya being used through the sasctl Session object and creates the appropriate
        score code and API calls required for the model and its associated content to
        be registered in SAS Model Manager.

        Generation of the score code requires that the `input_data`, `predict_method`,
        and `score_metrics` arguments are supplied. Otherwise, a warning will be
        generated stating that no score code is being created.

        The following are generated by this function if a path is provided in the
        model_files argument:

        * '\*Score.py'
            The Python score code file for the model.
        * '\*.zip'
            The zip archive of the relevant model files. In Viya 3.5 the Python
            score code is not present in this initial zip file.

        Parameters
        ----------
        model_files : str, pathlib.Path, or dict
            Either the directory location of the model files (string or Path object), or
            a dictionary containing the contents of all the model files.
        model_prefix : str
            The variable for the model name that is used when naming model files.
            (For example: 'hmeqClassTree + [Score.py || .pickle])'.
        project : str, dict, or RestObj
            The name or id of the model project, or a dictionary representation of the
            project.
        input_data : pandas.DataFrame, optional
            The `DataFrame` object contains the training data, and includes only the
            predictor columns. The write_score_code function currently supports int(64),
            float(64), and string data types for scoring. The default value is None.
        predict_method : Callable or list, optional
            The Python function used for model predictions and the expected output
            types. The expected output types can be passed as example values or as the
            value types. For example, if the model is a Scikit-Learn
            DecisionTreeClassifier, then pass either of the following:

            * [sklearn.tree.DecisionTreeClassifier.predict, ["A"]]
            * [sklearn.tree.DecisionTreeClassifier.predict_proba, [0.4, float]]

            The default value is None.
        score_metrics : list of str, optional
            The scoring score_metrics for the model. For classification models, it is
            assumed that the first value in the list represents the classification
            output. This function supports single and multi-class classification models.
            The default value is None.
        pickle_type : str, optional
            Indicator for the package used to serialize the model file to be uploaded to
            SAS Model Manager. The default value is `pickle`.
        project_version : str, optional
            The project version to import the model in to on SAS Model Manager. The
            default value is "latest".
        overwrite_model : bool, optional
            Set whether models with the same name should be overwritten when attempting
            to import the model. The default value is False.
        score_cas : bool, optional
            Sets whether models registered to SAS Viya 3.5 should be able to be scored
            and validated through both CAS and SAS Micro Analytic Service. If set to
            false, then the model will only be able to be scored and validated through
            SAS Micro Analytic Service. The default value is True.
        missing_values : bool, optional
            Sets whether data handled by the score code will impute for missing values.
            The default value is False.
        mlflow_details : dict, optional
            Model details from an MLFlow model. This dictionary is created by the
            read_mlflow_model_file function. The default value is None.
        predict_threshold : float, optional
            The prediction threshold for normalized probability score_metrics. Values
            are expected to be between 0 and 1. The default value is None.
        target_values : list of str, optional
            A list of target values for the target variable. This argument and the
            score_metrics argument dictate the handling of the predicted values from
            the prediction method. The order of the target values should reflect the
            order of the related probabilities in the model. The default value is None.
        overwrite_project_properties : bool, optional
            Set whether the project properties should be overwritten when attempting to
            import the model. The default value is False.
        target_index : int, optional
            Sets the index of success for a binary model. If target_values are given, this
            index should match the index of the target outcome in target_values. If target_values
            are not given, this index should indicate whether the the target probability variable
            is the first or second variable returned by the model. The default value is 1.
        **kwargs
            Other keyword arguments are passed to the following function:
            :meth:`.ScoreCode.write_score_code`


        Returns
        -------
        RestObj
            JSON response from the POST API call to SAS Model Manager for importing a
            zipped model
        model_files : dict, str, or pathlib.Path
            Dictionary representation of all files or the path the model files were
            generated from.
        """
        # For mlflow models, overwrite the provided or default pickle_type
        if mlflow_details:
            pickle_type = mlflow_details["serialization_format"]

        # Import model without generating score code (SAS Viya version invariant)
        if input_data is None or not predict_method or not score_metrics:
            warn(
                "The following arguments are required for the automatic generation of "
                "score code: input_data, predict_method, score_metrics."
            )
            if isinstance(model_files, dict):
                zip_io_file = zm.zip_files(model_files, model_prefix, is_viya4=False)
            else:
                zip_io_file = zm.zip_files(
                    Path(model_files), model_prefix, is_viya4=False
                )
                if cls.notebook_output:
                    print(f"All model files were zipped to {Path(model_files)}.")

            # Check if project name provided exists and raise an error or create a
            # new project
            project_response = mr.get_project(project)
            project = project_exists(
                project,
                project_response,
                target_values,
                model_files,
                overwrite_project_properties,
            )

            # Check if model with same name already exists in project.
            model_exists(
                project, model_prefix, overwrite_model, version_name=project_version
            )

            model = mr.import_model_from_zip(
                model_prefix, project, zip_io_file, version=project_version
            )
            return model, model_files
        # For SAS Viya 4, the score code can be written beforehand and imported with
        # all the model files
        elif current_session().version_info() == 4:
            score_code_dict = sc().write_score_code(
                model_prefix,
                input_data,
                predict_method,
                score_metrics=score_metrics,
                pickle_type=pickle_type,
                predict_threshold=predict_threshold,
                score_code_path=None if isinstance(model_files, dict) else model_files,
                target_values=target_values,
                missing_values=missing_values,
                score_cas=score_cas,
                target_index=target_index,
                preprocess_function=preprocess_function,
                **kwargs,
            )
            if score_code_dict:
                model_files.update(score_code_dict)
                zip_io_file = zm.zip_files(model_files, model_prefix, is_viya4=True)
            else:
                score_path = Path(model_files) / f"score_{model_prefix}.py"
                if cls.notebook_output:
                    print(
                        f"Model score code was written successfully to {score_path} and"
                        f" uploaded to SAS Model Manager."
                    )
                zip_io_file = zm.zip_files(
                    Path(model_files), model_prefix, is_viya4=True
                )
                if cls.notebook_output:
                    print(f"All model files were zipped to {Path(model_files)}.")

            # Check if project name provided exists and raise an error or create a
            # new project
            project_response = mr.get_project(project)
            project = project_exists(
                project,
                project_response,
                target_values,
                model_files,
                overwrite_project_properties,
            )

            # Check if model with same name already exists in project.
            model_exists(
                project, model_prefix, overwrite_model, version_name=project_version
            )

            model = mr.import_model_from_zip(
                model_prefix, project, zip_io_file, version=project_version
            )
            if cls.notebook_output:
                try:
                    print(
                        f"Model was successfully imported into SAS Model Manager as "
                        f"{model.name} with the following UUID: {model.id}."
                    )
                except AttributeError:
                    print("Model failed to import to SAS Model Manager.")

            return model, model_files
        # For SAS Viya 3.5, the score code is written after upload in order to know
        # the model UUID
        else:
            if isinstance(model_files, dict):
                zip_io_file = zm.zip_files(model_files, model_prefix, is_viya4=False)
            else:
                zip_io_file = zm.zip_files(
                    Path(model_files), model_prefix, is_viya4=False
                )
                if cls.notebook_output:
                    print(f"All model files were zipped to {Path(model_files)}.")

            # Check if project name provided exists and raise an error or create a
            # new project
            project_response = mr.get_project(project)
            project = project_exists(
                project,
                project_response,
                target_values,
                model_files,
                overwrite_project_properties,
            )

            # Check if model with same name already exists in project.
            model_exists(
                project, model_prefix, overwrite_model, version_name=project_version
            )

            model = mr.import_model_from_zip(
                model_prefix, project, zip_io_file, version=project_version
            )
            if cls.notebook_output:
                try:
                    print(
                        f"Model was successfully imported into SAS Model Manager as "
                        f"{model.name} with the following UUID: {model.id}."
                    )
                except AttributeError:
                    print("Model failed to import to SAS Model Manager.")

            score_code_dict = sc().write_score_code(
                model_prefix,
                input_data,
                predict_method,
                score_metrics=score_metrics,
                model=model,
                pickle_type=pickle_type,
                predict_threshold=predict_threshold,
                score_code_path=None if isinstance(model_files, dict) else model_files,
                target_values=target_values,
                missing_values=missing_values,
                score_cas=score_cas,
                target_index=target_index,
                preprocess_function=preprocess_function,
                **kwargs,
            )
            if score_code_dict:
                model_files.update(score_code_dict)
                return mr.get_model(model), model_files
            else:
                score_path = Path(model_files) / f"score_{model_prefix}.py"
                if cls.notebook_output:
                    print(
                        f"Model score code was written successfully to {score_path} and"
                        f" uploaded to SAS Model Manager."
                    )
                return mr.get_model(model), model_files
