import json

from requests import HTTPError

from .score_definitions import ScoreDefinitions
from .service import Service


class ScoreExecution(Service):
    """
    The Score Execution API is used to produce a score by
    executing the mapped code generated by score objects using the score definition.

    See Also
    --------
    `REST Documentation <https://developers.sas.com/rest-apis/scoreExecution-v2>`

    """

    _SERVICE_ROOT = "/scoreExecution"
    _score_definitions = ScoreDefinitions()

    (
        list_executions,
        get_execution,
        update_execution,
        delete_execution,
    ) = Service._crud_funcs("/executions", "execution")

    @classmethod
    def create_score_execution(
        cls,
        score_definition_id: str,
        description: str = "",
        output_server_name: str = "cas-shared-default",
        output_library_name: str = "Public",
        output_table_name: str = "",
    ):
        """Creates the score definition service.

        Parameters
        --------
        score_definition_id: str
            A score definition id representing score definition existing on the server that needs to be executed.
        description: str, optional
            Description of score execution. Defaults to an empty string.
        output_server_name: str, optional
            The name of the output server the output table and output library is stored in. Defaults to "cas-shared-default".
        output_library_name: str, optional
            The name of the output library the output table is stored in. Defaults to "Public".
        output_table_name: str, optional
            The name of the output table the score execution or analysis output will be stored in. Defaults to an empty string.

        Returns
        -------
        RestObj

        """

        # Gets information about the scoring object from the score definition and raises an exception if the score definition does not exist
        score_definition = cls._score_definitions.get_definition(score_definition_id)
        if score_definition.status_code >= 400:
            raise HTTPError(score_definition.json())
        score_exec_name = score_definition.get("name")
        model_uri = score_definition.get("objectDescriptor", ["uri"])
        model_name = score_definition.get("objectDescriptor", ["name"])
        model_input_library = score_definition.get("inputData", ["libraryName"])
        model_table_name = score_definition.get("inputData", ["tableName"])

        # Defining a default output table name if none is provided
        if not output_table_name:
            output_table_name = f"{model_name}_{score_definition_id}"

        # Getting all score executions that are using the inputted score_definition_id
       
        score_execution = cls.list_executions(filter=f"eq(scoreDefinitionId, '{score_definition_id}')")
        
        if score_execution.status_code >= 400:
            raise HTTPError(
                {
                    f"Something went wrong in the LIST_EXECUTIONS statement. See error: {score_execution.json()}"
                }
            )
        
        # Checking the count of the execution list to see if there are any score executions for this score_definition_id already running
        execution_count = score_execution.get("count")  # Exception catch location
        if execution_count == 1:
            execution_id = score_execution.get("items", [0], ["id"])
            deleted_execution = cls.delete_execution(execution_id)
            if deleted_execution.status_code >= 400:
                raise HTTPError(
                    {
                        f"Something went wrong in the DELETE statement. See error: {deleted_execution.json()}"
                    }
                )       

        headers_score_exec = {"Content-Type": "application/json"}

        create_score_exec = {
            "name": score_exec_name,
            "description": description,
            "hints": {
                "objectURI": model_uri,
                "inputTableName": model_table_name,
                "inputLibraryName": model_input_library,
            },
            "scoreDefinitionId": score_definition_id,
            "outputTable": {
                "tableName": output_table_name,
                "libraryName": output_library_name,
                "serverName": output_server_name,
            },
        }

        # Creating the score execution
        new_score_execution = cls.post(
            "scoreExecution/executions",
            data=json.dumps(create_score_exec),
            headers=headers_score_exec,
        )
        return new_score_execution
