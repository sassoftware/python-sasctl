import requests
from requests import HTTPError
import sys

from pathlib import Path
import json
from typing import Union, Optional

from ..core import current_session, delete, get, sasctl_command, RestObj
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
    _model_repository = ModelRepository()

    (
        list_definitions,
        get_definition,
        update_definition,
        delete_definition,
    ) = Service._crud_funcs("/definitions", "definition")

    @classmethod
    def create_score_definition(
        cls,
        score_def_name: str,
        model: Union[str, dict],
        table_name: str,
        use_cas_gateway: Optional[bool] = False,
        table_file: Union[str, Path] = None,
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
        model : str or dict
            The name or id of the model, or a dictionary representation of the model.
        table_name: str
            A user-inputted table name in CAS Management.
        use_cas_gateway: bool, optional
            Determines whether object uses CAS Gateway or not.
        table_file: str or Path, optional
            A user-provided path to an uploadable file. Defaults to None.
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
        # Changes object descriptor type to either use or not use CAS Gateway
        object_descriptor_type: str
        if use_cas_gateway:
            object_descriptor_type = "sas.models.model.python"
        else:
            object_descriptor_type = "sas.models.model.ds2"

        if cls._model_repository.is_uuid(model):
            model_id = model
        elif isinstance(model, dict) and "id" in model:
            model_id = model["id"]
        else:
            model = cls._model_repository.get_model(model)
            if not model:
                raise HTTPError(
                    "This model may not exist in a project or the model may not exist at all."
                )
            model_id = model["id"]

        model_project_id = model.get("projectId")
        model_project_version_id = model.get("projectVersionId")
        model_name = model.get("name")
        # Checking if the model exists and if it's in a project

        try:
            inputMapping = []
            for input_item in model.get("inputVariables"):
                var = {
                    "mappingValue": input_item["name"],
                    "mappingType": "datasource",
                    "variableName": input_item["name"],
                }
                inputMapping.append(var)

        except:
            print("This model does not have the optional 'inputVariables' parameter.")

        # Optional mapping - Maps the variables in the data to the variables of the score object. It's not necessary to create a score definition.

        table = cls._cas_management.get_table(table_name, library_name, server_name)
        if not table and not table_file:
            raise HTTPError(
                f"This table may not exist in CAS. Please include the `table_file` argument in the function call if it doesn't exist."
            )
        elif not table and table_file:
            cls._cas_management.upload_file(
                str(table_file), table_name
            )  # do I need to add a check if the file doesn't exist or does upload_file take care of that?
            table = cls._cas_management.get_table(table_name, library_name, server_name)
            if not table:
                raise HTTPError(
                    f"The file failed to upload properly or another error occurred."
                )
            # Checks if the inputted table exists, and if not, uploads a file to create a new table

        save_score_def = {
            "name": model_name,  # used to be score_def_name
            "description": description,
            "objectDescriptor": {
                "uri": f"/modelManagement/models/{model_id}",
                "name": f"{model_name}({model_version})",
                "type": f"{object_descriptor_type}",
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

        return cls.post(
            "/definitions", data=json.dumps(save_score_def), headers=headers_score_def
        )
        # The response information of the score definition can be seen as a JSON as well as a RestOBJ
