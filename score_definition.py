import requests

from pathlib import Path
import json
from typing import Union

from sasctl import Session
from sasctl._services.cas_management import CASManagement as cas

username = "edmdev"  # brmdev for a non-administrator session
password = "Go4thsas"

host = "edmlatest.ingress-nginx.edmtest-m1.edm.sashq-d.openstack.sas.com"  
#do we need to change host when putting it into the Git repository

sess = Session(host, username, password, protocol="http") 

DEFAULT_SERVER = "cas-shared-default"
DEFAULT_CASUSER = "Public"
DEFAULT_TABLE = ""

def score_definition(score_def_name: str,  model_id: str, table_name: str, description:str = "", server_name: str = DEFAULT_SERVER, library_name: str = DEFAULT_CASUSER, model_version: str = "latest"):
    """Creates the score definition service.
    
    Parameters
    --------
    score_def_name: str
        Name of score definition
    model_id: str
        A user-inputted model that exists in a project.
    table_name: str
        A user-inputted table name in CAS Management.
    description: str
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
    except:
        print("This model may not exist in a project.") #tested -> should also catch if model just doesn't exist


    #mapping
    inputMapping = []
    for i, input_item in enumerate(model_id["inputVariables"]):
        var = {"mappingValue":input_item["name"],"mappingType":"datasource","variableName":input_item["name"]}
        #mapping value and variable name are the same because the table values and model values are the same (remember MM quick start where automatically names matched up?)
        #will above always be the case? How do we handle exceptions

        inputMapping.append(var)
    
    try:
        table = sess.get(f"casManagement/servers/{server_name}/caslibs/{library_name}/tables/{table_name}")
        table_contents = table.json()
    except:
        table_data = input("The inputted table does not exist. Enter the file path for the data in your table (CSV, XLS, XLSX, SAS7BDT or SASHDAT file).")
        cas.upload_file(str(table_data), table_name)
        table = sess.get(f"casManagement/servers/{server_name}/caslibs/{library_name}/tables/{table_name}")

    save_score_def = {"name": score_def_name,
                "description": description,
                "objectDescriptor":{"uri":"/modelManagement/models/"+model_id,
                                    "name":model_id["name"] +"("+model_version+")",
                                    "type":"sas.models.model"}, 
                "inputData":{"type":"CASTable",
                             "serverName":DEFAULT_SERVER,
                             "libraryName":DEFAULT_CASUSER,
                             "tableName":table_name},
                "properties":{"tableBaseName":"",
                              "outputLibraryName":DEFAULT_CASUSER,
                              "outputTableName":score_def_name,
                              "modelOrigUri":"/modelRepository/models/"+ model_id["id"],
                              "projectUri":"/modelRepository/projects/"+ model_project_id,
                              "projectVersionUri": "/modelRepository/projects/"+ model_project_id+"/projectVersions/" + model_id["projectVersionId"],
                              "publishDestination":"",
                              "versionedModel": model_id["name"] +"("+ model_version+")"},
                "mappings": inputMapping} # Getting the input variables from the model that eventually have to be mapped to the input table created/retrieved from CAS Management
                                        #Array of mappings between Score Object variables and Input Data columns.
    
    headers_score_def = {
    'Content-Type': "application/json"
    }

    new_score_def = sess.post("scoreDefinitions/definitions", data=json.dumps(save_score_def), headers=headers_score_def)

    return new_score_def



