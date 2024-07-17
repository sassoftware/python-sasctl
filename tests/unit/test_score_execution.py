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


def test_create_score_execution():
    """
    Test Cases:
    - Valid score definition id?
        -yes
        -no
    - Valid execution id?

    -Valid count key?

    """

    with mock.patch(
        "sasctl._services.score_definitions.ScoreDefinitions.get_definition"
    ) as get_definition:
        with mock.patch(
            "sasctl._services.score_execution.ScoreExecution.get_execution"
        ) as get_execution:
            with mock.patch(
                "sasctl._services.score_execution.ScoreExecution.delete_execution"
            ) as delete_execution:
                pytest.skip()
                # raise HTTP error?
