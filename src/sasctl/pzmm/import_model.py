# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID
from warnings import warn

from .._services.model_repository import ModelRepository as mr
from ..core import current_session
from .write_score_code import ScoreCode as sc
from .zip_model import ZipModel as zm


def _create_project(project_name, model, repo, input_vars=None, output_vars=None):
    """Creates a project based on the model specifications.

    Parameters
    ----------
    project_name : str
        Name of the project to be created
    model : dict
        Model information
    repo : str or dict
        Repository in which to create the project
    input_vars : list
        Input variables formatted as {'name': 'varname'}
    output_vars
        Output variables formatted as {'name': 'varname'}

    Returns
    -------
    RestObj
        The created project
    """
    properties = {k: model[k] for k in model if k in ("function", "targetLevel")}

    function = model.get("function", "").lower()
    algorithm = model.get("algorithm", "").lower()

    # Get input & output variable lists
    # Note: copying lists to avoid altering original
    input_vars = input_vars or model.get("inputVariables", [])
    output_vars = output_vars or model.get("outputVariables", [])[:]
    input_vars = input_vars[:]
    output_vars = output_vars[:]

    # Set prediction or eventProbabilityVariable
    if output_vars:
        if function == "prediction":
            properties["predictionVariable"] = output_vars[0]["name"]
        else:
            properties["eventProbabilityVariable"] = output_vars[0]["name"]

    # Set targetLevel
    if properties.get("targetLevel") is None:
        if function == "classification" and "logistic" in algorithm:
            properties["targetLevel"] = "Binary"
        elif function == "prediction" and "regression" in algorithm:
            properties["targetLevel"] = "Interval"
        else:
            properties["targetLevel"] = None

    project = mr.create_project(
        project_name, repo, variables=input_vars + output_vars, **properties
    )

    # As of Viya 3.4 the 'predictionVariable' and 'eventProbabilityVariable'
    # parameters are not set during project creation.  Update the project if
    # necessary.
    needs_update = False
    for p in ("predictionVariable", "eventProbabilityVariable"):
        if project.get(p) != properties.get(p):
            project[p] = properties.get(p)
            needs_update = True

    if needs_update:
        project = mr.update_project(project)

    return project


def project_exists(response, project):
    """Checks if project exists on SAS Viya. If the project does not exist, then a new
    project is created or an error is raised.

    Parameters
    ----------
    response : dict
        JSON response of the get_project() call to model repository service.
    project : string or dict
        The name or id of the model project, or a dictionary representation of the project.

    Returns
    -------
    response : dict
        JSON response of the get_project() call to model repository service.

    Raises
    ------
    SystemError
        Alerts user that API calls cannot continue until a valid project is provided.
    """
    if response is None:
        try:
            warn("No project with the name or UUID {} was found.".format(project))
            UUID(project)
            raise SystemError(
                "The provided UUID does not match any projects found in SAS Model Manager. "
                + "Please enter a valid UUID or a new name for a project to be created."
            )
        except ValueError:
            repo = mr.default_repository().get("id")
            response = mr.create_project(project, repo)
            print("A new project named {} was created.".format(response.name))
            return response
    else:
        return response


def model_exists(project, name, force, versionName="latest"):
    """Checks if model already exists in the same project and either raises an error or deletes
    the redundant model. If no project version is provided, the version is assumed to be "latest".

    Parameters
    ----------
    project : str or dict
        The name or id of the model project, or a dictionary representation of the project.
    name : str or dict
        The name of the model.
    force : bool, optional
        Sets whether to overwrite models with the same name upon upload.
    versionName : str, optional
        Name of project version to check if a model of the same name already exists. Default
        value is "latest".

    Raises
    ------
    ValueError
        Model repository API cannot overwrite an already existing model with the upload model call.
        Alerts user of the force argument to allow multi-call API overwriting.
    """
    project = mr.get_project(project)
    projectId = project["id"]
    projectVersions = mr.list_project_versions(project)
    if versionName == "latest":
        modTime = [item["modified"] for item in projectVersions]
        latestVersion = modTime.index(max(modTime))
        versionId = projectVersions[latestVersion]["id"]
    else:
        for version in projectVersions:
            if versionName is version["name"]:
                versionId = version["id"]
                break
    projectModels = mr.get(
        "/projects/{}/projectVersions/{}/models".format(projectId, versionId)
    )

    for model in projectModels:
        # Throws a TypeError if only one model is in the project
        try:
            if model["name"] == name:
                if force:
                    mr.delete_model(model.id)
                else:
                    raise ValueError(
                        "A model with the same model name exists in project {}. Include the force=True argument to overwrite models with the same name.".format(
                            project.name
                        )
                    )
        except TypeError:
            if projectModels["name"] == name:
                if force:
                    mr.delete_model(projectModels.id)
                else:
                    raise ValueError(
                        "A model with the same model name exists in project {}. Include the force=True argument to overwrite models with the same name.".format(
                            project.name
                        )
                    )


class ImportModel:
    @classmethod
    def import_model(
        cls,
        model_dir,
        model_prefix,
        project,
        input_df,
        target_df,
        predict_method,
        metrics=None,
        project_version="latest",
        model_file_name=None,
        score_code_path=None,
        threshold=None,
        other_variable=False,
        is_h2o_model=False,
        force=False,
        binary_string=None,
        missing_values=False,
        mlflow_details=None,
    ):
        """Import model to SAS Model Manager using pzmm submodule.

        Using pzmm, generate Python score code and import the model files into
        SAS Model Manager. This function automatically checks the version of SAS
        Viya being used through the sasctl Session object and creates the appropriate
        score code and API calls required for the model and its associated content to
        be registered in SAS Model Manager.

        The following are generated by this function:
        * '*Score.py'
            The Python score code file for the model.
        * '*.zip'
            The zip archive of the relevant model files. In Viya 3.5 the Python score
            code is not present in this initial zip file.


        Parameters
        ----------
        model_dir : string or Path
            Directory location of the files to be zipped and imported as a model.
        model_prefix : string
            The variable for the model name that is used when naming model files.
            (For example: hmeqClassTree + [Score.py || .pickle]).
        project : str or dict
            The name or id of the model project, or a dictionary
            representation of the project.
        input_df : DataFrame
            The `DataFrame` object contains the training data, and includes only the 
            predictor columns. The write_score_code function currently supports int(64),
            float(64), and string data types for scoring.
        target_df : DataFrame
            The `DataFrame` object contains the training data for the target variable.
        predict_method : string
            User-defined prediction method for score testing. This should be
            in a form such that the model and data input can be added using
            the format() command.
            For example: '{}.predict_proba({})'.
        metrics : string list, optional
            The scoring metrics for the model. The default is a list of two
            metrics: EM_EVENTPROBABILITY and EM_CLASSIFICATION.
        project_version : string, optional
            The project version to import the model in to on SAS Model Manager. The 
            default value is "latest".
        model_file_name : string, optional
            Name of the model file that contains the model. Default is model_prefix + 
            '.pickle'.
        score_code_path : string, optional
            The local path of the score code file. Default is model_dir.
        threshold : float, optional
            The prediction threshold for probability metrics. For classification,
            below this threshold is a 0 and above is a 1.
        other_variable : boolean, optional
            The option for having a categorical other value for catching missing
            values or values not found in the training data set. The default setting
            is False.
        is_h2o_model : boolean, optional
            Sets whether the model is an H2O.ai Python model. Default is False.
        force : boolean, optional
            Sets whether to overwrite models with the same name upon upload. Default is 
            False.
        binary_string : string, optional
            Binary string representation of the model object. Default is None.
        missing_values : boolean, optional
            Sets whether data used for scoring needs to go through imputation for
            missing values before passed to the model. Default is False.
        mlflow_details : dict, optional
            Model details from an MLFlow model. This dictionary is created by the 
            read_mlflow_model_file function. Default is None.
        """
        # Set default output metrics
        if not metrics:
            metrics = ["EM_EVENTPROBABILITY", "EM_CLASSIFICATION"]

        # Initialize no score code or binary H2O model flags
        no_score_code = False
        binary_model = False

        if not mlflow_details:
            mlflow_details = {"serialization_format": "pickle"}

        if not score_code_path:
            score_code_path = Path(model_dir)
        else:
            score_code_path = Path(score_code_path)

        # Function to check for MOJO or binary model files in H2O models
        def get_files(extensions):
            all_files = []
            for ext in extensions:
                all_files.extend(score_code_path.glob(ext))
            return all_files

        # If the model file name is not provided, set a default value depending on
        # H2O and binary model status
        if not model_file_name:
            if is_h2o_model:
                binary_or_mojo = get_files(["*.mojo", "*.pickle"])
                if len(binary_or_mojo) == 0:
                    print(
                        "WARNING: An H2O model file was not found at {}. Score code "
                        "will not be automatically generated.".format(
                            str(score_code_path)
                        )
                    )
                    no_score_code = True
                elif len(binary_or_mojo) == 1:
                    if str(binary_or_mojo[0]).endswith(".pickle"):
                        binary_model = True
                        model_file_name = model_prefix + ".pickle"
                    else:
                        model_file_name = model_prefix + ".mojo"
                else:
                    print(
                        "WARNING: Both a MOJO and binary model file are present at {}."
                        "Score code will not be automatically generated.".format(
                            str(score_code_path)
                        )
                    )
                    no_score_code = True
            else:
                model_file_name = model_prefix + ".pickle"

        # Check the SAS Viya version number being used

        is_viya35 = current_session().version_info() == 3.5
        # For SAS Viya 4, the score code can be written beforehand and imported with
        # all of the model files
        if not is_viya35:
            if no_score_code:
                print("No score code was generated.")
            else:
                sc.write_score_code(
                    input_df,
                    target_df,
                    model_prefix,
                    predict_method,
                    model_file_name,
                    metrics=metrics,
                    score_code_path=score_code_path,
                    threshold=threshold,
                    other_variable=other_variable,
                    is_h2o_model=is_h2o_model,
                    is_binary_model=binary_model,
                    binary_string=binary_string,
                    missing_values=missing_values,
                    pickle_type=mlflow_details["serialization_format"],
                )
                print(
                    "Model score code was written successfully to {}.".format(
                        Path(score_code_path) / (model_prefix + "Score.py")
                    )
                )
            zip_io_file = zm.zip_files(Path(model_dir), model_prefix, is_viya4=True)
            print("All model files were zipped to {}.".format(Path(model_dir)))

            # Check if project name provided exists and raise an error or create a
            # new project
            project_response = mr.get_project(project)
            project = project_exists(project_response, project)

            # Check if model with same name already exists in project.
            model_exists(project, model_prefix, force, versionName=project_version)

            response = mr.import_model_from_zip(
                model_prefix, project, zip_io_file, version=project_version
            )
            try:
                print(
                    "Model was successfully imported into SAS Model Manager as {} "
                    "with UUID: {}.".format(
                        response.name, response.id
                    )
                )
            except AttributeError:
                print("Model failed to import to SAS Model Manager.")
        # For SAS Viya 3.5, the score code is written after upload in order to know
        # the model UUID
        else:
            zip_io_file = zm.zip_files(Path(model_dir), model_prefix, is_viya4=False)
            print("All model files were zipped to {}.".format(Path(model_dir)))

            # Check if project name provided exists and raise an error or create a
            # new project
            project_response = mr.get_project(project)
            project = project_exists(project_response, project)

            # Check if model with same name already exists in project.
            model_exists(project, model_prefix, force, versionName=project_version)

            response = mr.import_model_from_zip(
                model_prefix, project, zip_io_file, force, version=project_version
            )
            try:
                print(
                    "Model was successfully imported into SAS Model Manager as {} "
                    "with UUID: {}.".format(
                        response.name, response.id
                    )
                )
            except AttributeError:
                print("Model failed to import to SAS Model Manager.")
            if no_score_code:
                print("No score code was generated.")
            else:
                sc.write_score_code(
                    input_df,
                    target_df,
                    model_prefix,
                    predict_method,
                    model_file_name,
                    metrics=metrics,
                    score_code_path=score_code_path,
                    threshold=threshold,
                    other_variable=other_variable,
                    model=response.id,
                    is_h2o_model=is_h2o_model,
                    is_binary_model=binary_model,
                    binary_string=binary_string,
                    missing_values=missing_values,
                    pickle_type=mlflow_details["serialization_format"],
                )
                print(
                    "Model score code was written successfully to {} and uploaded to "
                    "SAS Model Manager".format(
                        Path(score_code_path) / (model_prefix + "Score.py")
                    )
                )
