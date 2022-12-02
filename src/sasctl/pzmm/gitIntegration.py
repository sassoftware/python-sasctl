# Copyright (c) 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID
from warnings import warn
import zipfile
import io

from .._services.model_repository import ModelRepository as mr
from ..core import RestObj

try:
    import git
    from git import Repo
except ImportError:
    git = None


def checkGitStatus():
    """Check to see if GitPython has been installed and if a valid git executable
    exists on the target system. If one of those two conditions is not met, then
    a RunTime error is raised.

    Raises
    ------
    RuntimeError
        Raised if an invalid git setup for git integration is detected.
    """
    if git is None:
        raise RuntimeError(
            "The 'GitPython' package and a valid git executable are required"
            + " for use of the git integration functions."
        )


def getZippedModel(model, gPath, project=None):
    """Retrieve a zipped file containing all of the model contents or a specified
    model. The project argument is only needed if the model argument is not a valid
    UUID or RestObj.

    Parameters
    ----------
    model : string or RestObj
        Model name, UUID, or RestObj which identifies the model. If only the model name
        is provided, the project name must also be supplied.
    gPath : string or Path
        Base directory of the git repository.
    project : string or RestObj, optional
        Project identifier, which is required when only the model name is supplied. Default
        is None.
    """
    params = {"format": "zip"}
    modelZip = mr.get("models/%s" % (model), params=params, format_="content")
    modelName = mr.get_model(model).name
    # Check project argument to determine project name
    if isinstance(project, RestObj):
        projectName = project.name
    elif project is None:
        projectName = mr.get_model(model).projectName
    else:
        projectName = mr.get_project(project).name
    # Check to see if project folder exists
    if (Path(gPath) / projectName).exists():
        # Check to see if model folder exists
        if (Path(gPath) / projectName / modelName).exists():
            with open(
                Path(gPath) / projectName / modelName / (modelName + ".zip"), "wb"
            ) as zFile:
                zFile.write(modelZip)
        else:
            newDir = Path(gPath) / projectName / modelName
            newDir.mkdir(parents=True, exist_ok=True)
            with open(
                Path(gPath) / projectName / modelName / (modelName + ".zip"), "wb"
            ) as zFile:
                zFile.write(modelZip)
    else:
        newDir = Path(gPath) / projectName
        newDir.mkdir(parents=True, exist_ok=True)
        newDir = Path(gPath) / projectName / modelName
        newDir.mkdir(parents=True, exist_ok=True)
        with open(
            Path(gPath) / (projectName + "/" + modelName + "/" + modelName + ".zip"),
            "wb",
        ) as zFile:
            zFile.write(modelZip)

    return modelName, projectName


def project_exists(response, project):
    """Checks if project exists on SAS Viya. If the project does not exist, then a new
    project is created or an error is raised.

    Parameters
    ----------
    response : RestObj
        JSON response of the get_project() call to model repository service.
    project : string or RestObj
        The name or id of the model project, or a RestObj representation of the project.

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


def model_exists(project, name, force):
    """Checks if model already exists and either raises an error or deletes the redundant model.

    Parameters
    ----------
    project : string or dict
        The name or id of the model project, or a dictionary representation of the project.
    name : str or dict
        The name of the model.
    force : bool, optional
        Sets whether to overwrite models with the same name upon upload.

    Raises
    ------
    ValueError
        Model repository API cannot overwrite an already existing model with the upload model call.
        Alerts user of the force argument to allow multi-call API overwriting.
    """
    project = mr.get_project(project)
    projectId = project["id"]
    projectModels = mr.get("/projects/{}/models".format(projectId))

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


class GitIntegrate:
    @classmethod
    def pullViyaModel(
        cls,
        model,
        gPath,
        project=None,
    ):
        """Send an API request in order to pull a model from a project in
        SAS Model Manager in a zipped format. The contents of the zip file
        include all files found in SAS Model Manager's model UI, except that
        read-only json files are updated to match the current state of the model.

        After pulling down the zipped model, unpack the file in the model folder.
        Overwrites files with the same name.

        If supplying a model name instead of model UUID, a project name or uuid must
        be supplied as well. Models in the model repository are allowed duplicate
        names, therefore we need a method of parsing the returned models.

        Parameters
        ----------
        model : string or RestObj
            A string or JSON response representing the model to be pulled down
        gPath : string or Path
            Base directory of the git repository.
        project : string or RestObj, optional
            A string or JSON response representing the project the model exists in, default is None.
        """
        # Try to pull down the model assuming a UUID or RestObj is provided
        try:
            if isinstance(model, RestObj):
                model = model.id
            else:
                UUID(model)
            projectName = mr.get_model(model).projectName
            modelName, projectName = getZippedModel(model, gPath, projectName)
        # If a name is provided instead, use the provided project name or UUID to find the correct model
        except ValueError:
            projectResponse = mr.get_project(project)
            if projectResponse is None:
                raise SystemError(
                    "For models with only a provided name, a project name or UUID must also be supplied."
                )
            projectName = projectResponse["name"]
            projectId = projectResponse["id"]
            projectModels = mr.get("/projects/{}/models".format(projectId))
            for model in projectModels:
                # Throws a TypeError if only one model is in the project
                try:
                    if model["name"] == model:
                        modelId = model.id
                        modelName, projectName = getZippedModel(
                            modelId, gPath, projectName
                        )
                except TypeError:
                    if projectModels["name"] == model:
                        modelId = projectModels.id
                        modelName, projectName = getZippedModel(
                            modelId, gPath, projectName
                        )

        # Unpack the pulled down zip model and overwrite any duplicate files
        mPath = Path(gPath) / "{projectName}/{modelName}".format(
            projectName=projectName, modelName=modelName
        )
        with zipfile.ZipFile(str(mPath / (modelName + ".zip")), mode="r") as zFile:
            zFile.extractall(str(mPath))

        # Delete the zip model objects in the directory to minimize confusion when uploading back to SAS Model Manager
        for zipFile in mPath.glob("*.zip"):
            zipFile.unlink()

    @classmethod
    def pushGitModel(
        cls, gPath, modelName=None, projectName=None, projectVersion="latest"
    ):
        """Push a single model in the git repository up to SAS Model Manager. This function
        creates an archive of all files in the directory and imports the zipped model.

        Parameters
        ----------
        gPath : string or Path
            Base directory of the git repository or path which includes project and model directories.
        modelName : string, optional
            Name of model to be imported, by default None
        projectName : string, optional
            Name of project the model is imported from, by default None
        projectVersion : str, optional
            Name of project version to import model in to. Default
            value is "latest".
        """
        if modelName is None and projectName is None:
            modelDir = gPath
            modelName = modelDir.name
            projectName = modelDir.parent.name
        else:
            modelDir = Path(gPath) / (projectName + "/" + modelName)
        for zipFile in modelDir.glob("*.zip"):
            zipFile.unlink()
        fileNames = []
        fileNames.extend(sorted(Path(modelDir).glob("*")))
        with zipfile.ZipFile(
            str(modelDir / (modelDir.name + ".zip")), mode="w"
        ) as zFile:
            for file in fileNames:
                zFile.write(str(file), arcname=file.name)
        with open(modelDir / (modelDir.name + ".zip"), "rb") as zFile:
            zipIOFile = io.BytesIO(zFile.read())
            # Check to see if provided project argument is a valid project on SAS Model Manager
            projectResponse = mr.get_project(projectName)
            project = project_exists(projectResponse, projectName)
            projectName = project.name
            # Check if model with same name already exists in project. Delete if it exists.
            model_exists(projectName, modelName, True)
            mr.import_model_from_zip(
                modelName, projectName, zipIOFile, projectVersion=projectVersion
            )

    @classmethod
    def gitRepoPush(cls, gPath, commitMessage, remote="origin", branch="main"):
        """Create a new commit with new files, then push changes from the local repository to a remote
        branch. The default remote branch is origin.

        Parameters
        ----------
        gPath : string or Path
            Base directory of the git repository.
        commitMessage : string
            Commit message for the new commit
        remote : str, optional
            Remote name for the remote repository, by default 'origin'
        branch : string
            Branch name for the target pull branch from remote, by default 'main'
        """
        checkGitStatus()

        repo = Repo(gPath)
        repo.git.add(all=True)
        repo.index.commit(commitMessage)
        repo.git.push(remote, branch)

    @classmethod
    def gitRepoPull(cls, gPath, remote="origin", branch="main"):
        """Pull down any changes from a remote branch of the git repository. The default branch is
        origin.

        Parameters
        ----------
        gPath : string or Path
            Base directory of the git repository.
        remote : string
            Remote name for the remote repository, by default 'origin'
        branch : string
            Branch name for the target pull branch from remote, by default 'main'
        """
        checkGitStatus()

        repo = git.Git(gPath)
        repo.pull(remote, branch)

    @classmethod
    def pushGitProject(cls, gPath, project=None):
        """Using a user provided project name, search for the project in the specified git repository,
        check if the project already exists on SAS Model Manager (create a new project if it does not),
        then upload each model found in the git project to SAS Model Manager

        Parameters
        ----------
        gPath : string or Path
            Base directory of the git repository or the project directory.
        project : string or RestObj
            Project name, UUID, or JSON response from SAS Model Manager.
        """
        # Check to see if provided project argument is a valid project on SAS Model Manager
        projectResponse = mr.get_project(project)
        project = project_exists(projectResponse, project)
        projectName = project.name

        # Check if project exists in git path and produce an error if it does not
        pPath = Path(gPath) / projectName
        if pPath.exists():
            models = [x for x in pPath.glob("*") if x.is_dir()]
            if len(models) == 0:
                print("No models were found in project {}.".format(projectName))
            print(
                "{numModels} models were found in project {projectName}.".format(
                    numModels=len(models), projectName=projectName
                )
            )
        else:
            raise FileNotFoundError(
                "No directory with the name {} was found in the specified git path.".format(
                    project
                )
            )

        # Loop through paths of models and upload each to SAS Model Manager
        for model in models:
            # Remove any extra zip objects in the directory
            for zipFile in model.glob("*.zip"):
                zipFile.unlink()
            cls.pushGitModel(model)

    @classmethod
    def pullMMProject(cls, gPath, project):
        """Following the user provided project argument, pull down all models from the
        corresponding SAS Model Manager project into the mapped git directories.

        Parameters
        ----------
        gPath : string or Path
            Base directory of the git repository.
        project : string or RestObj
            The name or id of the model project, or a RestObj representation of the project.
        """
        # Check to see if provided project argument is a valid project on SAS Model Manager
        projectResponse = mr.get_project(project)
        project = project_exists(projectResponse, project)
        projectName = project.name
        # Check if project exists in git path and create it if it does not
        pPath = Path(gPath) / projectName
        if not pPath.exists():
            Path(pPath).mkdir(parents=True, exist_ok=True)

        # Return a list of model names from SAS Model Manager project
        modelResponse = mr.get("projects/{}/models".format(project.id))
        if modelResponse == []:
            raise FileNotFoundError(
                "No models were found in the specified project. A new project folder "
                + "has been created if it did not already exist within the git repository."
            )
        modelNames = []
        modelId = []
        for model in modelResponse:
            modelNames.append(model.name)
            modelId.append(model.id)
        # For each model, search for an appropriate model directory in the project directory and pull down the model
        for name, id in zip(modelNames, modelId):
            mPath = pPath / name
            # If the model directory does not exist, create one in the project directory
            if not mPath.exists():
                Path(mPath).mkdir(parents=True, exist_ok=True)
            cls.pullViyaModel(id, mPath.parents[1])
