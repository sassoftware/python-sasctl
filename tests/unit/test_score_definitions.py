#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2024, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from requests import HTTPError

import sasctl.pzmm as pzmm
from sasctl import current_session
from sasctl.core import RestObj, VersionInfo, request
from sasctl._services.score_definitions import ScoreDefinitions as sd

def test_create_score_definition():
    """
    Test Cases:
    - Valid model id?
        - yes
        - no
    - Valid input mapping?
        - yes 
        - no
    -Valid table name?
        -yes
        -no
    -Valid file?
        -yes
        -no
    """
    
    with mock.patch(
        "sasctl._services.model_repository.ModelRepository.get_model"
    ) as get_model:
         with mock.patch(
            "sasctl._services.cas_management.CASManagement.get_table"
        ) as get_table:
            with mock.patch(
                "sasctl._services.cas_management.CASManagement.upload_file"
            ) as upload_file:
                get_model.return_value.status_code = 404
                with pytest.raises(HTTPError):
                    sd.create_score_definition(
                        score_def_name = "test_create_sd",
                        model_id = "12345",
                        table_name = "test_table"
                    )
                # get_model.return_value.status_code = 200
                # get_model.return_value.json.return_value = {
                #     "id": "12345",
                #     "projectId": "54321",
                #     "projectVersionId": "67890",
                #     "name": "test_model"
                #     "inputVariables": {"A", "B", "C"} #look at syntax combine square brackets and curly bracket
                # }
                #     sd.create_score_definition(
                #         score_def_name = "test_create_sd",
                #         model_id = "12345",
                #         table_name = "test_table",
                #         mappings = {
                #             {"mappingVariable": "A", "datasource": "datasource", "variableName": "A"},
                #             {"mappingVariable": "B", "datasource": "datasource", "variableName": "B"},
                #             {"mappingVariable": "C", "datasource": "datasource", "variableName": "C"}
                #         }
                #     )
                    
                get_model.return_value.status_code = 200
                get_model.return_value.json.return_value = {
                    "id": "12345",
                    "projectId": "54321",
                    "projectVersionId": "67890",
                    "name": "test_model",
                }
                get_table.return_value.status_code = 404
                with pytest.raises(HTTPError):
                    sd.create_score_definition(
                            score_def_name = "test_create_sd",
                            model_id = "12345",
                            table_name = "test_table"
                    )
                get_table.return_value.status_code = 404
                upload_file.return_value = None
                get_table.return_value.status_code = 404
                with pytest.raises(HTTPError):
                    sd.create_score_definition(
                            score_def_name = "test_create_sd",
                            model_id = "12345",
                            table_name = "test_table",
                            table_file = "test_path"
                        )
                
                get_table.return_value.status_code = 404
                upload_file.return_value = RestObj
                get_table.return_value.status_code = 200
                response = sd.create_score_definition(
                        score_def_name = "test_create_sd",
                        model_id = "12345",
                        table_name = "test_table",
                        table_file = "test_path"
                )
                assert response
                
                
                
                
                # get_table.return_value.status_code = 200
                # get_table.return_value.json.return_value = {
                #     "tableName": "test_table"
                # }
                # sd.create_score_definition(
                #     score_def_name = "test_create_sd",
                #     model_id = "12345",
                #     table_name = "test_table"
                # )

                
                
                
                
         
                # # Test for valid model id, no input mapping, get_table succeeded, and post succeeds
               
                # get_table.return_value = RestObj()
                # raise_for_status.return_value = None
                
                
                
                # assert response
                
                # Test for invalid model id
                #get_model.return_value = RestObj()
                #with pytest.raises(
                    #HTTPError
                #): #test tget_Table since we don't know what the return value is restobj
                    #sd.create_score_definition(
                        #score_def_name = "test_create_sd",
                        #model_id = "12345",
                        #table_name = "test_table",
                    #)
                #Test for valid and invalid for each
                #Start with testing invalid and realize you can work from the last call to the first because the last call working means the first call must have worked
               
    