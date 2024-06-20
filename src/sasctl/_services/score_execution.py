import requests

from pathlib import Path
import json
from typing import Union

from sasctl import Session
from sasctl._services.cas_management import CASManagement as cas


def score_execution(score_definition, description: str = "", output_server_name: str = "cas-shared-default", output_library_name: str = "Public", output_table_name: str = ""):
    """Creates the score definition service.
    
    Parameters
    --------
    score_definition
        Score Definition RestObj.
    description: str
        Description of score definition. Defaults to an empty string.
    output_server_name: str, optional
        The server within CAS that the output table is in. Defaults to "cas-shared-default".
    output_library_name: str, optional
        The library within the CAS server the output table exists in. Defaults to "Public".
    output_table_name: str, optional
        The name of the output table. Defaults to an empty string.
     
    
    Returns
    -------
    RestObj

    """
    try:
        score_definition_id = score_definition.json()["id"]
        score_definition_name = score_definition.json()["name"]
        score_definition_model_id = score_definition.json["modelId"]
        score_definition_model_name = score_definition.json["modelName"]
    except:
        print("The input score definition object may not exist. Please check it was passed in correctly")

    if output_table_name == "":
        output_table_name = f"{score_definition_model_name}_{score_definition_id}"

    try:
        filtered_score_executions = sess.get(f"scoreExecution/executions?filter=eq(scoreExecutionRequest.scoreDefinitionId,%27{score_definition_id}%27)")
        deleted_score_execution = sess.delete(f"scoreExecution/executions/{filtered_score_executions["items"][0]["id"]}")
    #can there be more than one score execution for a score definition running (execCount >= 1)
    except:
        pass

    import requests

from pathlib import Path
import json
from typing import Union

from sasctl import Session
from sasctl._services.cas_management import CASManagement as cas




def score_execution(score_definition,  description: str = "", output_server_name: str = "cas-shared-default", output_library_name: str = "Public", output_table_name: str = ""):
    """Creates the score definition service.
     
    Parameters
    --------
    score_definition: RestObj
        A score definition existing on the server.
    description: str
        Description of score execution. Defaults to an empty string.
    output_server_name: str
        The name of the output server the output table and output library is stored in. Defaults to "cas-shared-default".
    output_library_name: str
        The name of the output library the output table is stored in. Defaults to "Public".
    output_table_name: str
        The name of the output table the score execution or analysis output will be stored in. Defaults to an empty string.
    
    Returns
    -------
    RestObj

    """

    headers_score_exec = {
    "Content-Type": "application/json"
    }

    try:

        score_def_id = score_definition.json()["id"]
        score_exec_name = score_definition.json()["name"]
        model_uri = score_definition.json()["objectDescriptor"]["uri"]
        model_name = score_definition.json()["objectDescriptor"]["name"]
        model_input_library = score_definition.json()["inputData"]["libraryName"]
        model_input_server = score_definition.json()["inputData"]["serverName"]
        model_table_name = score_definition.json()["inputData"]["tableName"]
        
        model_table = sess.get(f"casManagement/servers/{model_input_server}/caslibs/{model_input_library}/tables/{model_table_name}")
        model_table_name = model_table.json()["name"]
        model_table_id = model_table.json()["id"]
    except:
        print("The score definition may not exist or the fields are incomplete.")

    if output_table_name == "":
        output_table_name = f"{model_name}_{score_def_id}" #is output table the same as I specified in score_definition/do I need to create one or does it automatically create one?
    try:
        score_execution = sess.get(f"/scoreExecution/executions?filter=eq(scoreExecutionRequest.scoreDefinitionId,%27{score_def_id}%27)") #getting all of them because 
        execCount = score_execution.json()['count']
        if execCount == 1:
            execTaskId = score_execution.json()["items"][0]["id"]
            deleted_score_execution = sess.delete(f"scoreExecution/executions/{execTaskId}")
            print(deleted_score_execution)
    except:
        print("There may not be a score execution already running.")

        create_score_exec = {"name": score_exec_name,
                "description": description,
                "hints": {"objectURI":model_uri, #because I'm scoring a model, is that why this is required?
                          "modelTable":model_table_id,
                          "inputTableName":model_table_name,
                          "inputLibraryName": model_input_library},
                "scoreDefinitionId": score_def_id,
                "outputTable": {"tableName": output_table_name,
                                "libraryName": output_library_name,
                                "serverName": output_server_name}}
    
    new_score_execution = sess.post("scoreExecution/executions", data=json.dumps(create_score_exec), header=headers_score_exec)
    return new_score_execution            




