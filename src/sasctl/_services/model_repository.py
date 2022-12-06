#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Model Repository service supports registering and managing models."""

import datetime
from warnings import warn

from .service import Service
from ..core import current_session, get, delete, sasctl_command, HTTPError

FUNCTIONS = {
    "Analytical",
    "Classification",
    "Clustering",
    "Forecasting",
    "Prediction",
    "Text categorization",
    "Text extraction",
    "Text sentiment",
    "Text topics",
    "Sentiment",
}


class ModelRepository(Service):
    """Implements the Model Repository REST API.

    The Model Repository API provides support for registering, organizing,
    and managing models within a common model repository.

    See Also
    --------
    `REST Documentation <https://developer.sas.com/apis/rest/DecisionManagement
    /#model-repository>`_

    """

    _SERVICE_ROOT = "/modelRepository"

    list_repositories, _, update_repository, delete_repository = Service._crud_funcs(
        "/repositories", "repository"
    )

    list_projects, get_project, update_project, delete_project = Service._crud_funcs(
        "/projects", "project"
    )

    list_models, get_model, update_model, delete_model = Service._crud_funcs(
        "/models", "model"
    )

    @classmethod
    def get_astore(cls, model):
        """Get the ASTORE for a model registered in the model repository.

        Parameters
        ----------
        model :  str or dict
            The name or id of the model, or a dictionary representation of the
            model.

        Returns
        ----------
        binary?

        """
        # TODO: Download binary object?

        link = cls.get_model_link(model, "analyticStore", refresh=True)

        if link is not None:
            return link.get("href")

    @classmethod
    def get_model_link(cls, model, rel, refresh=False):
        """Retrieve a link from a model's set of links.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the
            model.
        rel : str
            The name of the link to retrieve
        refresh : bool, optional
            Whether the model's data should be refreshed before searching for
            the link.

        Returns
        -------
        dict or None
            Dictionary containing the link's properties

        """
        # Try to find the link if the model is a dictionary
        if isinstance(model, dict):
            link = cls.get_link(model, rel)
            if link is not None or not refresh:
                return link

        # Model either wasn't a dictionary or it didn't include the link
        # Pull the model data and then check for the link
        model = cls.get_model(model, refresh=refresh)

        return cls.get_link(model, rel)

    @classmethod
    def get_score_code(cls, model):
        """Get the score code for a model registered in the model repository.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model.

        Returns
        -------
        str
            The code designated as the model's score code

        """
        link = cls.get_model_link(model, "scoreCode", refresh=True)

        if link is not None:
            scorecode_uri = link.get("href")
            return get(
                scorecode_uri,
                headers={"Accept": "text/vnd.sas.models.score.code.ds2package"},
            )

    @classmethod
    def get_model_contents(cls, model):
        """Retrieve the additional files and data associated with the model.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the
            model.

        Returns
        -------
        list
            The code designated as the model's score code

        """

        link = cls.get_model_link(model, "contents", refresh=True)
        contents = cls.request_link(link, "contents")

        # By default, request_link() will unwrap a length-1 list.
        # If that happens, re-wrap so a list is always returned.
        if isinstance(contents, list):
            return contents

        return [contents]

    @classmethod
    @sasctl_command("get", "repositories")
    def get_repository(cls, repository, refresh=False):
        """Return a repository instance.

        Parameters
        ----------
        repository : str or dict
            Name, ID, or dictionary representation of the repository.
        refresh : bool, optional
            Obtain an updated copy of the repository.

        Returns
        -------
        RestObj or None
            A dictionary containing the repository attributes or None.

        Notes
        -------
        If `repository` is a complete representation of the repository it will be
        returned unless `refresh` is set.  This prevents unnecessary REST
        calls when data is already available on the client.

        """
        # If the input already appears to be the requested object just
        # return it, unless a refresh of the data was explicitly requested.
        if isinstance(repository, dict) and all(
            k in repository for k in ("id", "name")
        ):
            if refresh:
                repository = repository["id"]
            else:
                return repository

        if cls.is_uuid(repository):
            try:
                # Attempt to GET the repository directly.  Access may be restricted, so allow HTTP 403 errors
                # and fall back to using list_repositories() instead.
                return cls.get("/repositories/{id}".format(id=repository))
            except HTTPError as e:
                if e.code != 403:
                    raise e

        results = cls.list_repositories()

        # Not sure why, but as of 19w04 the filter doesn't seem to work
        for result in results:
            if result["name"] == str(repository) or result["id"] == str(repository):
                # Make a request for the specific object so that ETag
                # is included, allowing updates.
                try:
                    if cls.get_link(result, "self"):
                        return cls.request_link(result, "self")

                    id_ = result.get("id", result["name"])
                    return cls.get("/repositories/{id}".format(id=id_))
                except HTTPError as e:
                    # NOTE: As of Viya 4.0.1 access to GET a repository is restricted to admin users out of the box.
                    # Try to GET the repository, but ignore any 403 (permission denied) errors.
                    if e.code != 403:
                        raise e
                return result

    @classmethod
    def create_model(
        cls,
        model,
        project,
        description=None,
        modeler=None,
        function=None,
        algorithm=None,
        tool=None,
        score_code_type=None,
        training_table=None,
        event_prob_variable=None,
        event_target_value=None,
        is_champion=False,
        is_challenger=False,
        location=None,
        target_variable=None,
        is_retrainable=False,
        is_immutable=False,
        properties=None,
        input_variables=None,
        output_variables=None,
    ):
        """Create a model in an existing project or folder.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the
            model.
        project : str or dict
            The name or id of the model project, or a dictionary representation
            of the model project.
        description : str, optional
            The description of the model.
        modeler : str, optional
            Name of the user that created the model.  Current user name will be
            used if unspecified.
        function : str, optional
            The function of the model, valid values include: analytical,
            classification, cluster, forecasting, prediction, Text analytics,
            transformation.
        algorithm : str, optional
            The name of the model algorithm.
        tool : str, optional
            The name of the model tool, can be 'Python 2' or 'Python 3'.
        score_code_type : str, optional
            The score code type for the model.
        training_table : str, optional
            The train data table.
        event_prob_variable : str, optional
            The name of the event probability variable.  Used for
            classification models only.
        event_target_value : str, optional
            The target event value.  Used for classification models only.
        is_champion : bool, optional
            Indicates whether the model should be designated as the
            project's champion model.  Defaults to False.
        is_challenger : bool, optional
            Indicates whether the model should be designated as a
            challenger model in the project.  Defaults to False.
        location : str, optional,
            The location of this model.
        target_variable : str, optional
            The name of the target variable.
        is_retrainable : bool
            Indicates whether the model can be retrained or not.
        is_immutable : bool
            Indicates whether the model can be changed or not.
        properties : dict, optional
            Custom model properties provided as name: value pairs.
            Allowed types are int, float, string, datetime.date, and datetime.datetime
        input_variables : array_like, optional
            Model input variables. By default, these are the same as the model
            project.
        output_variables : array_like, optional
            Model output variables. By default, these are the same as the model
             project.
        project_version : str
            Name of project version to import model in to. Default
            value is "latest".

        Returns
        -------
        str
            The model schema returned in JSON format.

        """
        properties = properties or {}

        if isinstance(model, str):
            model = {"name": model}

        p = cls.get_project(project)
        if p is None:
            raise ValueError("Unable to find project '%s'" % project)

        # Use any explicitly passed parameter value first.
        # Fall back to values in the model dict.
        model["projectId"] = p["id"]
        model["modeler"] = modeler or model.get("modeler") or current_session().username
        model["description"] = description or model.get("description")
        model["function"] = function or model.get("function")
        model["algorithm"] = algorithm or model.get("algorithm")
        model["tool"] = tool or model.get("tool")
        model["champion"] = is_champion or model.get("champion")

        if is_champion:
            model["role"] = "champion"
        elif is_challenger:
            model["role"] = "challenger"

        model.setdefault("properties", [])
        for k, v in properties.items():
            if type(v) in (int, float):
                t = "numeric"
            elif type(v) is datetime.date:
                # NOTE: do not use isinstance() to compare types as isinstance(v, datetime.date) evaluates to True
                #       even for datetime.datetime instances.
                # Convert to datetime to extract timestamp and then scale to milliseconds
                v = datetime.datetime(v.year, v.month, v.day).timestamp()
                v = int(v * 1000)
                t = "date"
            elif type(v) is datetime.datetime:
                # Extract timestamp and scale to milliseconds
                v = int(v.timestamp() * 1000)
                t = "dateTime"
            else:
                t = "string"
                v = str(v)

            model["properties"].append({"name": k, "value": v, "type": t})

        model["scoreCodeType"] = score_code_type or model.get("scoreCodeType")
        model["trainTable"] = training_table or model.get("trainTable")
        model[
            "classificationEventProbabilityVariableName"
        ] = event_prob_variable or model.get(
            "classificationEventProbabilityVariableName"
        )
        model["classificationTargetEventValue"] = event_target_value or model.get(
            "classificationTargetEventValue"
        )
        model["location"] = location or model.get("location")
        model["targetVariable"] = target_variable or model.get("targetVariable")
        model["retrainable"] = is_retrainable or model.get("retrainable")
        model["immutable"] = is_immutable or model.get("immutable")
        model["inputVariables"] = input_variables or model.get("inputVariables", [])
        model["outputVariables"] = output_variables or model.get("outputVariables", [])
        model["version"] = 2

        return cls.post(
            "/models",
            json=model,
            headers={"Content-Type": "application/vnd.sas.models.model+json"},
        )

    @classmethod
    def add_model_content(
        cls, model, file, name, role=None, content_type="multipart/form-data"
    ):
        """Add additional files to the model.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model.
        file : str, dict, or bytes
            A file related to the model, such as the model code.
        name : str
            Name of the file related to the model.
        role : str
            Role of the model file, such as 'Python pickle'. Default value is None.
        content_type : str
            An HTTP Content-Type value. Default value is multipart/form-data.

        Returns
        -------
        str
            The model content schema.

        """
        if cls.is_uuid(model):
            id_ = model
        elif isinstance(model, dict) and "id" in model:
            id_ = model["id"]
        else:
            model = cls.get_model(model)
            id_ = model["id"]

        if content_type == "multipart/form-data" and isinstance(file, bytes):
            content_type = "application/octet-stream"
        elif isinstance(file, dict):
            import json

            file = json.dumps(file)

        files = {"files": (name, file, content_type)}

        if role is None:
            params = {}
        else:
            params = {"role": role}
        params = "&".join("{}={}".format(k, v) for k, v in params.items())

        # If the file already exists, a 409 error will be returned
        try:
            return cls.post(
                "/models/{}/contents".format(id_), files=files, params=params
            )
        # Deletes the duplicate content and reruns the API call
        except HTTPError as e:
            if e.code == 409:
                model_contents = cls.get_model_contents(id_)
                for item in model_contents:
                    if item.name == name:
                        cls.delete("/models/{}/contents/{}".format(id_, item.id))

                        # Return json stream to beginning of file content
                        if hasattr(files["files"][1], "seek"):
                            files["files"][1].seek(0)

                        return cls.post(
                            "/models/{}/contents".format(id_),
                            files=files,
                            params=params,
                        )
            else:
                raise e

    @classmethod
    def default_repository(cls):
        """Get the built-in default repository.

        Returns
        -------
        RestObj

        """
        all_repos = cls.list_repositories()

        if all_repos:
            # If nothing else, return the first repository
            repo = all_repos[0]

            # Check repository names to find a better default.
            # 'Repository 1' was default in 19w04
            # 'Public' was default in 19w21
            for r in all_repos:
                if r.name in ("Repository 1", "Public"):
                    repo = r
                    break
            return repo

    @classmethod
    def create_project(cls, project, repository, **kwargs):
        """Create a model project in the given model repository.

        Parameters
        ----------
        project : str or dict
            The name or id of the model project, or a dictionary representation
            of the project.
        repository : str or dict
            The name or id of the model repository, or a dictionary
            representation of the repository.

        Returns
        -------
        RestObj

        """
        if isinstance(project, str):
            project = {"name": project}

        repository = cls.get_repository(repository)

        project["repositoryId"] = repository["id"]
        project["folderId"] = repository["folderId"]

        project.update(kwargs)
        return cls.post(
            "/projects",
            json=project,
            headers={"Content-Type": "application/vnd.sas.models.project+json"},
        )

    @classmethod
    def import_model_from_zip(
        cls, name, project, file, description=None, version="latest"
    ):
        """Import a model and contents as a ZIP file into a model project.

        Parameters
        ----------
        name : str or dict
            The name of the model.
        project : str or dict
            The name or id of the model project, or a dictionary
            representation of the project.
        file : bytes
            The ZIP file containing the model and contents.
        description : str
            The description of the model.
        version : str, optional
            Name of the project version. Default value is "latest".

        Returns
        -------
        RestObj
            The API response after importing the model.

        """
        project_info = cls.get_project(project)

        if project_info is None:
            raise ValueError("Project `%s` could not be found." % str(project))

        params = {
            "name": name,
            "description": description,
            "type": "ZIP",
            "projectId": project_info.id,
            "versionOption": version,
        }
        params = "&".join("{}={}".format(k, v) for k, v in params.items())

        r = cls.post(
            "/models#octetStream",
            data=file.read(),
            params=params,
            headers={"Content-Type": "application/octet-stream"},
        )
        return r

    @classmethod
    def create_model_version(cls, model, minor=False):
        """Create a new version of an existing model.

        Create a new major (X.0) or minor (1.X) version of an existing
        model.  All contents from the current version are copied to the new
        version.

        Parameters
        ----------
        model : str or dict
            The name, id, or dictionary representation of a model.
        minor : bool, optional
            Whether the new version should be a minor version increment from
            the current version.  Otherwise, a new major version will be
            created.  Defaults to False.

        Returns
        -------
        RestObj
            The new version of the model.

        """
        model_obj = cls.get_model(model)
        if cls.get_model_link(model_obj, "addModelVersion") is None:
            raise ValueError("Unable to create a new version for model '%s'" % model)

        option = "minor" if minor else "major"

        # As of 8/9/19 addModelVersion returns the CURRENT model version,
        # not the NEW version.  Refresh the model to get the latest version.
        cls.request_link(model_obj, "addModelVersion", json={"option": option})

        return cls.get_model(model_obj, refresh=True)

    @classmethod
    def list_model_versions(cls, model):
        """Get a list of previous versions of a model.

        The current model version is not included in the results.

        Parameters
        ----------
        model : str or dict
            The name, id, or dictionary representation of a model.

        Returns
        -------
        list

        """
        model = cls.get_model(model)
        if cls.get_model_link(model, "modelVersions") is None:
            raise ValueError("Unable to retrieve versions for model '%s'" % model)

        return cls.request_link(model, "modelVersions")

    @classmethod
    def copy_analytic_store(cls, model):
        """Copy model ASTOREs to a pre-defined server location.

        Copies all of the analytic stores for a model to the pre-defined
        server location (/config/data/modelsvr/astore).
        To enable publishing and scoring, models that contain analytic stores
        need the ASTORE files to be copied to a set location
        (/config/data/modelsrv/astore).  This location is used for
        integration with Event Stream Processing and others. This request
        invokes an asynchronous call to copy the analytic store files. Check
        the individual analytic store uris to get the completion state:
        pending, copying, success, failure. Please review the full Model
        Manager documentation before using.

        Parameters
        ----------
        model : str or dict
            The name, id, or dictionary representation of a model.

        Returns
        -------
        RestObj

        """
        rel = "copyAnalyticStore"

        model = cls.get_model(model)

        if cls.get_link(model, rel) is None:
            model = cls.get_model(model, refresh=True)

        return cls.request_link(model, "copyAnalyticStore")

    @classmethod
    def delete_model_contents(cls, model):
        """Deletes all contents (files) in the model.

        Parameters
        ----------
        model : str or dict
            The name, id, or dictionary representation of a model.

        Returns
        -------

        """
        rel = "delete"

        filelist = cls.get_model_contents(model)
        for delfile in filelist:
            modelfileuri = cls.get_link(delfile, rel)
            delete(modelfileuri["uri"])

    @classmethod
    def copy_python_resources(cls, model):
        """Moves a model's score resources to the Compute server.

        Copies all of the Python score resources for a model to the pre-defined
        server location (/models/resources/viya/<model-UUID>/). To enable
        publishing and scoring, models that contain Python scoring resources
        need the score resource files to be copied to a set location
        (/models/resources/viya/<model-UUID>/). This location is used for
        integration with Event Stream Processing and others. This request
        invokes an asynchronous call to copy the score resource files. Check
        the individual score resource uris to get the completion state:
        pending, copying, success, failure. Please review the full Model
        Manager documentation before using.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model.

        Returns
        -------
        RestObj or None
            JSON response detailing the API metadata

        Warns
        -----
        UserWarning
            If no score resources exist for the model.

        """

        model_obj = cls.get_model(model)

        if model_obj is None:
            raise ValueError("No model '{}' found.".format(model))

        # Check if symbolic link for resource directories exists
        try:
            response = cls.put(
                "/models/%s/scoreResources" % model_obj["id"],
                headers={"Accept": "application/json"},
            )
            if not response:
                warn("No score resources found for model '{}'".format(model_obj.name))
            return response
        except HTTPError as e:
            if e.code == 406:
                raise OSError(
                    "The SAS Viya system you are attempting to move score resources with requires an additional"
                    + " administrator action in order to complete. Please see the documentation at "
                    + "https://go.documentation.sas.com/doc/en/calcdc/3.5/calmodels/n10916nn7yro46n119nev9sb912c.htm,"
                    + "which details the corollary approach for configuring analytic store model files."
                )
            else:
                raise e

    @classmethod
    def convert_python_to_ds2(cls, model):
        """Converts a Python model to DS2

        For SAS Viya 3.5 Python models on SAS Model Manager, wrap the Python score code in DS2
        and convert the model score code type to DS2. Models converted in this way are not
        scoreable by CAS.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model.

        Returns
        -------
        API response
            JSON response detailing the API metadata

        """
        if cls.is_uuid(model):
            id_ = model
        elif isinstance(model, dict) and "id" in model:
            id_ = model["id"]
        else:
            model = cls.get_model(model)
            id_ = model["id"]

        if isinstance(model, (str, dict)):
            model = cls.get_model(id_)

        ETag = model._headers["ETag"]
        accept = "text/vnd.sas.source.ds2"
        content = "application/json"

        return cls.put(
            "/models/%s/typeConversion" % id_,
            headers={"Accept-Item": accept, "Content-Type": content, "If-Match": ETag},
        )

    @classmethod
    def get_model_details(cls, model):
        """Get model details from SAS Model Manager

        Get model details that pertain to model properties, model metadata,
        model input, output, and target variables, and user-defined values.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model.

        Returns
        -------
        API response
            JSON response detailing the model details

        """
        if cls.is_uuid(model):
            id_ = model
        elif isinstance(model, dict) and "id" in model:
            id_ = model["id"]
        else:
            model = cls.get_model(model)
            id_ = model["id"]

        return cls.get("/models/%s" % id_)

    @classmethod
    def list_project_versions(cls, project):
        """Get a list of all versions of a project.

        Parameters
        ----------
        project : str or dict
            The name or id of the model project, or a dictionary representation
            of the model project.

        Returns
        -------
        list of dicts
            List of dicts representing different project versions. Dict key/value
            pairs are as follows.
                name : str
                id : str
                number : str
                modified : datetime

        """
        project_info = cls.get_project(project)

        if project_info is None:
            raise ValueError("Project `%s` could not be found." % str(project))

        projectVersions = cls.get(
            "/projects/{}/projectVersions".format(project_info.id)
        )
        versionList = []
        try:
            for version in projectVersions:
                versionDict = {
                    "name": version.name,
                    "id": version.id,
                    "number": version.versionNumber,
                    "modified": datetime.datetime.strptime(
                        version.modifiedTimeStamp, "%Y-%m-%dT%H:%M:%S.%fZ"
                    ),
                }
                versionList.append(versionDict)
        except AttributeError:
            versionDict = {
                "name": projectVersions.name,
                "id": projectVersions.id,
                "number": projectVersions.versionNumber,
                "modified": datetime.datetime.strptime(
                    projectVersions.modifiedTimeStamp, "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
            }
            versionList.append(versionDict)
        return versionList
