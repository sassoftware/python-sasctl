# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID
from warnings import warn

from ..core import current_session
from .._services.model_repository import ModelRepository as mr
from .writeScoreCode import ScoreCode as sc
from .zip_model import ZipModel as zm


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
    def pzmmImportModel(
        cls,
        zPath,
        modelPrefix,
        project,
        inputDF,
        targetDF,
        predictmethod,
        metrics=None,
        projectVersion="latest",
        modelFileName=None,
        pyPath=None,
        threshPrediction=None,
        otherVariable=False,
        isH2OModel=False,
        force=False,
        binaryString=None,
        missingValues=False,
        mlFlowDetails=None,
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
        zPath : string or Path
            Directory location of the files to be zipped and imported as a model.
        modelPrefix : string
            The variable for the model name that is used when naming model files.
            (For example: hmeqClassTree + [Score.py || .pickle]).
        project : str or dict
            The name or id of the model project, or a dictionary
            representation of the project.
        inputDF : DataFrame
            The `DataFrame` object contains the training data, and includes only the predictor
            columns. The writeScoreCode function currently supports int(64), float(64),
            and string data types for scoring.
        targetDF : DataFrame
            The `DataFrame` object contains the training data for the target variable.
        predictMethod : string
            User-defined prediction method for score testing. This should be
            in a form such that the model and data input can be added using
            the format() command.
            For example: '{}.predict_proba({})'.
        metrics : string list, optional
            The scoring metrics for the model. The default is a list of two
            metrics: EM_EVENTPROBABILITY and EM_CLASSIFICATION.
        projectVersion : string, optional
            The project version to import the model in to on SAS Model Manager. The default value
            is latest.
        modelFileName : string, optional
            Name of the model file that contains the model. By default None and assigned as
            modelPrefix + '.pickle'.
        pyPath : string, optional
            The local path of the score code file. By default None and assigned as the zPath.
        threshPrediction : float, optional
            The prediction threshold for probability metrics. For classification,
            below this threshold is a 0 and above is a 1.
        otherVariable : boolean, optional
            The option for having a categorical other value for catching missing
            values or values not found in the training data set. The default setting
            is False.
        isH2OModel : boolean, optional
            Sets whether the model is an H2O.ai Python model. By default False.
        force : boolean, optional
            Sets whether to overwrite models with the same name upon upload. By default False.
        binaryString : string, optional
            Binary string representation of the model object. By default None.
        missingValues : boolean, optional
            Sets whether data used for scoring needs to go through imputation for
            missing values before passed to the model. By default False.
        mlFlowDetails : dict, optional
            Model details from an MLFlow model. This dictionary is created by the readMLModelFile function.
            By default None.
        """
        # Set metrics internal to function call if no value is given
        if metrics is None:
            metrics = ["EM_EVENTPROBABILITY", "EM_CLASSIFICATION"]

        # Initialize no score code or binary H2O model flags
        noScoreCode = False
        binaryModel = False

        if mlFlowDetails is None:
            mlFlowDetails = {"serialization_format": "pickle"}

        if pyPath is None:
            pyPath = Path(zPath)
        else:
            pyPath = Path(pyPath)

        # Function to check for MOJO or binary model files in H2O models
        def getFiles(extensions):
            allFiles = []
            for ext in extensions:
                allFiles.extend(pyPath.glob(ext))
            return allFiles

        # If the model file name is not provided, set a default value depending on H2O and binary model status
        if modelFileName is None:
            if isH2OModel:
                binaryOrMOJO = getFiles(["*.mojo", "*.pickle"])
                if len(binaryOrMOJO) == 0:
                    print(
                        "WARNING: An H2O model file was not found at {}. Score code will not be automatically generated.".format(
                            str(pyPath)
                        )
                    )
                    noScoreCode = True
                elif len(binaryOrMOJO) == 1:
                    if str(binaryOrMOJO[0]).endswith(".pickle"):
                        binaryModel = True
                        modelFileName = modelPrefix + ".pickle"
                    else:
                        modelFileName = modelPrefix + ".mojo"
                else:
                    print(
                        "WARNING: Both a MOJO and binary model file are present at {}. Score code will not be automatically generated.".format(
                            str(pyPath)
                        )
                    )
                    noScoreCode = True
            else:
                modelFileName = modelPrefix + ".pickle"

        # Check the SAS Viya version number being used

        isViya35 = current_session().version_info() == 3.5
        # For SAS Viya 4, the score code can be written beforehand and imported with all of the model files
        if not isViya35:
            if noScoreCode:
                print("No score code was generated.")
            else:
                sc.writeScoreCode(
                    inputDF,
                    targetDF,
                    modelPrefix,
                    predictmethod,
                    modelFileName,
                    metrics=metrics,
                    pyPath=pyPath,
                    threshPrediction=threshPrediction,
                    otherVariable=otherVariable,
                    isH2OModel=isH2OModel,
                    isBinaryModel=binaryModel,
                    binaryString=binaryString,
                    missingValues=missingValues,
                    pickleType=mlFlowDetails["serialization_format"],
                )
                print(
                    "Model score code was written successfully to {}.".format(
                        Path(pyPath) / (modelPrefix + "Score.py")
                    )
                )
            zipIOFile = zm.zip_files(Path(zPath), modelPrefix, is_viya4=True)
            print("All model files were zipped to {}.".format(Path(zPath)))

            # Check if project name provided exists and raise an error or create a new project
            projectResponse = mr.get_project(project)
            project = project_exists(projectResponse, project)

            # Check if model with same name already exists in project.
            model_exists(project, modelPrefix, force, versionName=projectVersion)

            response = mr.import_model_from_zip(
                modelPrefix, project, zipIOFile, version=projectVersion
            )
            try:
                print(
                    "Model was successfully imported into SAS Model Manager as {} with UUID: {}.".format(
                        response.name, response.id
                    )
                )
            except AttributeError:
                print("Model failed to import to SAS Model Manager.")
        # For SAS Viya 3.5, the score code is written after upload in order to know the model UUID
        else:
            zipIOFile = zm.zip_files(Path(zPath), modelPrefix, is_viya4=False)
            print("All model files were zipped to {}.".format(Path(zPath)))

            # Check if project name provided exists and raise an error or create a new project
            projectResponse = mr.get_project(project)
            project = project_exists(projectResponse, project)

            # Check if model with same name already exists in project.
            model_exists(project, modelPrefix, force, versionName=projectVersion)

            response = mr.import_model_from_zip(
                modelPrefix, project, zipIOFile, force, projectVersion=projectVersion
            )
            try:
                print(
                    "Model was successfully imported into SAS Model Manager as {} with UUID: {}.".format(
                        response.name, response.id
                    )
                )
            except AttributeError:
                print("Model failed to import to SAS Model Manager.")
            if noScoreCode:
                print("No score code was generated.")
            else:
                sc.writeScoreCode(
                    inputDF,
                    targetDF,
                    modelPrefix,
                    predictmethod,
                    modelFileName,
                    metrics=metrics,
                    pyPath=pyPath,
                    threshPrediction=threshPrediction,
                    otherVariable=otherVariable,
                    model=response.id,
                    isH2OModel=isH2OModel,
                    isBinaryModel=binaryModel,
                    binaryString=binaryString,
                    missingValues=missingValues,
                    pickleType=mlFlowDetails["serialization_format"],
                )
                print(
                    "Model score code was written successfully to {} and uploaded to SAS Model Manager".format(
                        Path(pyPath) / (modelPrefix + "Score.py")
                    )
                )
