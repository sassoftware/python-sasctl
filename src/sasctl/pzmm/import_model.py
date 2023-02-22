# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID
from warnings import warn
from typing import Union, Optional, Callable, List

from pandas import DataFrame

from .._services.model_repository import ModelRepository as mr
from ..core import current_session, RestObj, PagedList
from ..utils.misc import check_if_jupyter
from .write_score_code import ScoreCode as sc
from .zip_model import ZipModel as zm

# TODO: add converter for any type of dataset (list, dataframe, numpy array)


def project_exists(
    response: Union[dict, RestObj], project: Union[str, dict, RestObj]
) -> RestObj:
    """
    Checks if project exists on SAS Viya. If the project does not exist, then a new
    project is created or an error is raised.

    Parameters
    ----------
    response : RestObj or dict
        JSON response of the get_project() call to model repository service.
    project : str, dict, or RestObj
        The name or id of the model project, or a dictionary representation of the
        project.

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
            # TODO: implement _create_project() call from tasks.py
            response = mr.create_project(project, repo)
            print(f"A new project named {response.name} was created.")
            return response
    else:
        return response


def model_exists(
    project: Union[str, dict, RestObj],
    name: str,
    force: bool,
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
        Sets whether to overwrite models with the same name upon upload.
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
        input_data: DataFrame,
        predict_method: Callable[..., List],
        output_variables: List[str],
        project: Union[str, dict, RestObj],
        pickle_type: str = "pickle",
        project_version: str = "latest",
        missing_values: bool = False,
        overwrite_model: bool = False,
        score_cas: bool = True,
        mlflow_details: Optional[dict] = None,
        predict_threshold: Optional[float] = None,
        target_values: Optional[List[str]] = None,
        **kwargs,
    ):
        """Import model to SAS Model Manager using pzmm submodule.

        Using pzmm, generate Python score code and import the model files into
        SAS Model Manager. This function automatically checks the version of SAS
        Viya being used through the sasctl Session object and creates the appropriate
        score code and API calls required for the model and its associated content to
        be registered in SAS Model Manager.

        The following are generated by this function if a path is provided in the
        model_files argument:
        * '*Score.py'
            The Python score code file for the model.
        * '*.zip'
            The zip archive of the relevant model files. In Viya 3.5 the Python score
            code is not present in this initial zip file.


        Parameters
        ----------
        model_files : string, Path, or dict
            Either the directory location of the model files (string or Path object), or
            a dictionary containing the contents of all the model files.
        model_prefix : string
            The variable for the model name that is used when naming model files.
            (For example: hmeqClassTree + [Score.py || .pickle]).
        input_data : pandas.DataFrame
            The `DataFrame` object contains the training data, and includes only the
            predictor columns. The write_score_code function currently supports int(64),
            float(64), and string data types for scoring.
        predict_method : Python function
            The Python function used for model predictions. For example, if the model is
            a Scikit-Learn DecisionTreeClassifier, then pass either of the following:
            sklearn.tree.DecisionTreeClassifier.predict
            sklearn.tree.DecisionTreeClassifier.predict_proba
        output_variables : string list
            The scoring output_variables for the model. For classification models, it is
             assumed that the first value in the list represents the classification
            output. This function supports single and multi-class classification models.
        project : str, dict, or RestObj
            The name or id of the model project, or a dictionary representation of the
            project.
        pickle_type : string, optional
            Indicator for the package used to serialize the model file to be uploaded to
            SAS Model Manager. The default value is `pickle`.
        project_version : string, optional
            The project version to import the model in to on SAS Model Manager. The
            default value is "latest".
        overwrite_model : bool, optional
            Set whether models with the same name should be overwritten when attempting
            to import the model. The default value is False.
        score_cas : boolean, optional
            Sets whether models registered to SAS Viya 3.5 should be able to be scored
            and validated through both CAS and SAS Micro Analytic Service. If set to
            false, then the model will only be able to be scored and validated through
            SAS Micro Analytic Service. The default value is True.
        missing_values : boolean, optional
            Sets whether data handled by the score code will impute for missing values.
            The default value is False.
        mlflow_details : dict, optional
            Model details from an MLFlow model. This dictionary is created by the
            read_mlflow_model_file function. The default value is None.
        predict_threshold : float, optional
            The prediction threshold for normalized probability output_variables. Values
             are expected to be between 0 and 1. The default value is None.
        target_values : list of strings, optional
            A list of target values for the target variable. This argument and the
            output_variables argument dictate the handling of the predicted values from
            the prediction method. The default value is None.
        kwargs : dict, optional
            Other keyword arguments are passed to the following function:
            * sasctl.pzmm.ScoreCode.write_score_code(...,
                binary_h2o_model=False,
                binary_string=None,
                model_file_name=None,
                mojo_model=False,
                statsmodels_model=False
            )
        """
        # For mlflow models, overwrite the provided or default pickle_type
        if mlflow_details:
            pickle_type = mlflow_details["serialization_format"]

        # For SAS Viya 4, the score code can be written beforehand and imported with
        # all the model files
        if current_session().version_info() == 4:
            score_code_dict = sc.write_score_code(
                model_prefix,
                input_data,
                predict_method,
                output_variables,
                pickle_type=pickle_type,
                predict_threshold=predict_threshold,
                score_code_path=None if isinstance(model_files, dict) else model_files,
                target_values=target_values,
                missing_values=missing_values,
                score_cas=score_cas,
                **kwargs,
            )
            if score_code_dict:
                model_files.update(score_code_dict)
                zip_io_file = zm.zip_files(model_files, model_prefix, is_viya4=True)
            else:
                score_path = Path(model_files) / (model_prefix + "Score.py")
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
            project = project_exists(project_response, project)

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

            if score_code_dict:
                return model_files
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
            project = project_exists(project_response, project)

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

            score_code_dict = sc.write_score_code(
                model_prefix,
                input_data,
                predict_method,
                output_variables,
                model=model,
                pickle_type=pickle_type,
                predict_threshold=predict_threshold,
                score_code_path=None if isinstance(model_files, dict) else model_files,
                target_values=target_values,
                missing_values=missing_values,
                score_cas=score_cas,
                **kwargs,
            )
            if score_code_dict:
                model_files.update(score_code_dict)
                return model_files
            else:
                score_path = Path(model_files) / (model_prefix + "Score.py")
                if cls.notebook_output:
                    print(
                        f"Model score code was written successfully to {score_path} and"
                        f" uploaded to SAS Model Manager."
                    )
