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
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from requests import HTTPError

import sasctl.pzmm as pzmm
from sasctl import current_session
from sasctl.core import RestObj, VersionInfo
from sasctl._services.score_definitions import ScoreDefinitions as sd
from sasctl._services.score_execution import ScoreExecution as se


def test_create_score_execution():
    """
    Test Cases:
    - Valid score definition id?
        -yes
        -no
    - Valid execution id?

    -Valid count key? -> treated like input mapping -> no because i think it's required
    - output table -> treat like input mapping but within the create_score_execution step (do I for library and server in score definition thought? but I think you need a target step here too)

    """
    with mock.patch("sasctl.core.Session._get_authorization_token"):
        current_session("example.com", "username", "password")    

    with mock.patch(
        "sasctl._services.score_definitions.ScoreDefinitions.get_definition"
    ) as get_definition:
        with mock.patch(
            "sasctl._services.score_execution.ScoreExecution.list_executions"
        ) as list_executions:
            with mock.patch(
                "sasctl._services.score_execution.ScoreExecution.delete_execution"
            ) as delete_execution:
                with mock.patch(
                   "sasctl._services.score_execution.ScoreExecution.post" 
                ) as post:
                    get_definition.return_value.status_code = 404
                    with pytest.raises(HTTPError):
                        se.create_score_execution(
                            score_definition_id="12345"
                        )
                    get_definition.return_value.status_code = 200
                    get_definition.return_value.json.return_value = {
                        "inputData": {
                            "libraryName": "cas-shared-default",
                            "tableName": ""
                        },
                        "name": "score_def_name",
                        "objectDescriptor": {
                            "name": "test_model",
                            "type": "sas.publish.example",
                            "uri": "/modelPublish/models/example"
                         }
                    }
                    list_executions.return_value.status_code = 400 #test case worked with .return_value.status_code but caused an HTTP error that said error with list_Executions
                    with pytest.raises(HTTPError):
                        se.create_score_execution(
                            score_definition_id="12345"
                        )

                    list_executions.status_code = 200 #why does this not have a return value and how is this related to different score_Defintion instantiation in score_execution
                    list_executions.return_value.json.return_value = {"count": 1}
                    delete_execution.return_value.status_code = 404
                    with pytest.raises(HTTPError):
                        se.create_score_execution(
                            score_definition_id="12345"
                        )
                    #Test cases that don't work or I haven't tested
                    delete_execution.return_value.status_code = 200 
                    response = se.create_score_execution(
                        score_definition_id="3456"
                    )
                    assert response

                    list_executions.status_code = 200 #shouldn't this work with 200 response and not give me a type rror >= since it's not running 400 statement
                    
                    list_executions.return_value.json.return_value = {"count": 0}
                    response = se.create_score_execution(
                        score_definition_id="12345"
                    )
                    assert response

                    list_executions.status_code = 200
                    list_executions.return_value.json.return_value = {"count": 0}
                    response = se.create_score_execution(
                        score_definition_id="12345",
                        output_table_name = "test_output_table"
                    )
                    assert response #target needed here to test with and without specified output_table_name
                    
                
                #pytest.skip()
                # raise HTTP error?
#notes -> how to test count because it should only delete if count == 1 so how do I test that if/else
#notes -> output table case, again TARGET?