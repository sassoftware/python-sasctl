import requests

from pathlib import Path
import json
from typing import Union

from sasctl import Session
from sasctl._services.cas_management import CASManagement as cas

DEFAULT_SERVER = "cas-shared-default"
DEFAULT_CASUSER = "Public"
DEFAULT_TABLE = ""

def score_execution(score_def_name: str,  model_id: str, table_name: str, description:str = "", server_name: str = DEFAULT_SERVER, library_name: str = DEFAULT_CASUSER, model_version: str = "latest"):
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
    
    all_score_executions = sess.get("scoreExecution/executions")
