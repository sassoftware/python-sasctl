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
from sasctl.core import RestObj, VersionInfo
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
                # # Test for valid model id, no input mapping, get_table succeeded, and post succeeds
                # get_model.return_value = RestObj(
                #         projectId = "54321",
                #         projectVersionId = "67890",
                #         name = "test_model",
                #     )
                # get_table.return_value = RestObj()
                # raise_for_status.return_value = None
                
                # response = sd.create_score_definition(
                #     score_def_name = "test_create_sd",
                #     model_id = "12345",
                #     table_name = "test_table",
                # )
                
                # assert response
                
                # Test for invalid model id
                get_model.return_value = RestObj()
                with pytest.raises(
                    HTTPError
                ):
                    sd.create_score_definition(
                        score_def_name = "test_create_sd",
                        model_id = "12345",
                        table_name = "test_table",
                    )
                    
    