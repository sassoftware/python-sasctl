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

    
            




