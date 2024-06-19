import requests

from pathlib import Path
import json
from typing import Union

from sasctl import Session
from sasctl._services.cas_management import CASManagement as cas

DEFAULT_SERVER = "cas-shared-default"
DEFAULT_CASUSER = "Public"
DEFAULT_TABLE = ""

def score_definition(score_def_name: str,  model_id: str, table_name: str, description:str = "", server_name: str = "cas-shared-default", library_name: str = "Public", model_version: str = "latest"):
    """Creates the score definition service.
    
    Parameters
    --------
    score_def_name: str
        Name of score definition.
    model_id: str
        A user-inputted model id where the model exists in a project.
    table_name: str
        A user-inputted table name in CAS Management.
    description: str, optional
        Description of score definition. Defaults to an empty string.
    server_name: str, optional
        The server within CAS that the table is in. Defaults to "cas-shared-default"
    library_name: str, optional
        The library within the CAS server the table exists in. Defaults to "Public"
    model_version: str, optional
        Defaults to "latest". 
    
    Returns
    -------
    RestObj

    """

    try:
        model = sess.get(f"modelRepository/models/{model_id}")
        model_project_id = model.json()["projectId"]
        model_project_version_id = model.json()["projectVersionId"]
        model_name = model.json()["name"]
        
    except:
        print("This model may not exist in a project.")

    #mapping
    inputMapping = []
    for input_item in model.json()["inputVariables"]:
        var = {"mappingValue":input_item["name"],"mappingType":"datasource","variableName":input_item["name"]}
        inputMapping.append(var)
    
    table = sess.get(f"casManagement/servers/{server_name}/caslibs/{library_name}/tables/{table_name}")
    try:
        print(table.json()) #because table_name is an empty string/none, it's ignore the last part of the uri and just printing all the tables, that's why error code 400 is not being recognized
        table_status_code = table.json()['httpStatusCode']
        if table_status_code == 404 or table_status_code == 400:
            table_file = input("The inputted table does not exist. Enter the file path for the data in your table (CSV, XLS, XLSX, SAS7BDT or SASHDAT file).")
            cas.upload_file(str(table_file), table_name)
            table = sess.get(f"casManagement/servers/{server_name}/caslibs/{library_name}/tables/{table_name}")
    except:
        print("Table GET API call successful due to no 'httpStatusCode' key.")