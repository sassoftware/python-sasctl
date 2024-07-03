import requests
from requests import HTTPError
import sys

from pathlib import Path
import json
from typing import Union

from ..core import current_session, delete, get, sasctl_command
from .cas_management import CASManagement
from .model_repository import ModelRepository
from .service import Service


class ScoreDefinitions(Service):
    """
    Used for creating and maintaining score definitions.

    The Score Definitions API is used for creating and maintaining score definitions.

    See Also
    --------
    `REST Documentation <https://developers.sas.com/rest-apis/scoreDefinitions-v3>`

    """

    _SERVICE_ROOT = "/scoreDefinitions"
    _cas_management = CASManagement()
    _model_respository = ModelRepository()

    list_definitions, get_definition, update_definition, delete_definition = (
        Service._crud_funcs("/definitions", "definition")
    )

    @classmethod
    def create_score_definition(
        cls,
        score_def_name: str,
        model_id: str,
        table_name: str,
        description: str = "",
        server_name: str = "cas-shared-default",
        library_name: str = "Public",
        model_version: str = "latest",
    ):
        """Creates the score definition service.

        Parameters
        --------
        score_def_name: str
            Name of score definition.
        model_id: str
            A user-inputted model if where the model exists in a project.
        table_name: str
            A user-inputted table name in CAS Management.
        description: str, optional
            Description of score definition. Defaults to an empty string.
        server_name: str, optional
            The server within CAS that the table is in. Defaults to "cas-shared-default".
        library_name: str, optional
            The library within the CAS server the table exists in. Defaults to "Public".
        model_version: str, optional
            The user-chosen version of the model with the specified model_id. Defaults to "latest".

        Returns
        -------
        RestObj

        """

        try:
            model = cls._model_respository.get_model(model_id)
            model_project_id = model.json()["projectId"]
            model_project_version_id = model.json()["projectVersionId"]
            model_name = model.json()["name"]

        except KeyError:
            raise HTTPError(
                "This model may not exist in a project or the model may not exist at all. See error: " + model.json()
            )
        # Checking if the model exists and if it's in a project

        try:
            inputMapping = []
            for input_item in model.json()["inputVariables"]:
                var = {
                    "mappingValue": input_item["name"],
                    "mappingType": "datasource",
                    "variableName": input_item["name"],
                }
                inputMapping.append(var)
        except:
            print("This model does not have the optional 'inputVariables' parameter.")

        # Optional mapping - Maps the variables in the data to the variables of the score object. It's not necessary to create a score definition.

        table = cls._cas_management.get_table(server_name, library_name, table_name)
        try:
            table.raise_for_status()
        except requests.exceptions.HTTPError as error:
            print(f"HTTP Error: {error}")
            table_file = input(
                "The inputted table does not exist. Enter the file path for the data in your table (CSV, XLS, XLSX, SAS7BDT or SASHDAT file)."
            )
            cls._cas_management.upload_file(
                str(table_file), table_name
            )  # do I need to add a check if the file doesn't exist or does upload_file take care of that?
            table = cls._cas_management.get_table(server_name, library_name, table_name)
            # Checks if the inputted table exists, and if not, uploads a file to create a new table
        try:
            table.raise_for_status()
        except requests.exceptions.HTTPError as error:
            print(f"HTTP Error: {error}")
            raise Exception(
                "Something went wrong when creating a table. Check to see if the file path inputted exists."
            )

        save_score_def = {
            "name": score_def_name,
            "description": description,
            "objectDescriptor": {
                "uri": f"/modelManagement/models/{model_id}",
                "name": f"{model_name}({model_version})",
                "type": "sas.models.model",
            },
            "inputData": {
                "type": "CASTable",
                "serverName": server_name,
                "libraryName": library_name,
                "tableName": table_name,
            },
            "properties": {
                "tableBaseName": "",
                "modelOrigUri": f"/modelRepository/models/{model_id}",
                "projectUri": f"/modelRepository/projects/{model_project_id}",
                "projectVersionUri": f"/modelRepository/projects/{model_project_id}/projectVersions/{model_project_version_id}",
                "publishDestination": "",
                "versionedModel": f"{model_name}({model_version})",
            },
            "mappings": inputMapping,
        }
        # Consolidating all of the model and table information to create the score definition information

        headers_score_def = {"Content-Type": "application/json"}

        new_score_def = cls.post(
            "/definitions", data=json.dumps(save_score_def), headers=headers_score_def
        )

        return new_score_def
        # The response information of the score definition can be seen as a JSON as well as the RestOBJ
