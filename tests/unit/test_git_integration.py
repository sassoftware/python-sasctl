#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import pickle
import random
import sys
import tempfile
import unittest
import uuid
import warnings
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sasctl.pzmm as pzmm
from sasctl.core import RestObj
from sasctl.pzmm import GitIntegrate as GI


@pytest.fixture()
def get_zipped_model_mocks():
    with patch.multiple(
        "sasctl._services.model_repository.ModelRepository",
        get=DEFAULT,
        get_model=DEFAULT,
        get_project=DEFAULT,
        list_models=DEFAULT,
    ) as mocks:
        yield mocks


def test_check_git_status():
    """
    Test Cases:
    - GitPython not installed
    """
    from sasctl.pzmm.git_integration import check_git_status

    with patch("sasctl.pzmm.git_integration.git", None):
        with pytest.raises(RuntimeError):
            check_git_status()


def test_get_zipped_model(tmp_path_factory, get_zipped_model_mocks):
    """
    Test cases:
    - model
        - RestObj
        - UUID
        - name
            - without project
            - with project
    - git project
        - exists
            - model exists
            - model DNE
        - DNE
    """
    from sasctl.pzmm.git_integration import get_zipped_model

    get_zipped_model_mocks["get_model"].return_value = RestObj(
        {"name": "mtest", "id": "123abc", "projectName": "ptest"}
    )
    get_zipped_model_mocks["get_project"].return_value = RestObj({"name": "ptest"})
    get_zipped_model_mocks["get"].return_value = bytes(b"789xyz")
    get_zipped_model_mocks["list_models"].return_value = [
        RestObj({"name": "mtest", "id": "123abc", "projectName": "ptest"})
    ]

    # RestObj model + git repo with project/model
    tmp_dir = Path(tmp_path_factory.mktemp("test1"))
    (tmp_dir / "ptest").mkdir()
    (tmp_dir / "ptest" / "mtest").mkdir()
    model, project = get_zipped_model(
        RestObj({"name": "mtest", "id": "123abc", "projectName": "ptest"}), tmp_dir
    )
    assert ("mtest", "ptest") == (model, project)
    assert Path(tmp_dir / project / model / (model + ".zip")).exists()

    # UUID model + git repo with project
    tmp_dir = Path(tmp_path_factory.mktemp("test2"))
    (tmp_dir / "ptest").mkdir()
    model, project = get_zipped_model(str(uuid.uuid4()), tmp_dir)
    assert ("mtest", "ptest") == (model, project)
    assert Path(tmp_dir / project / model / (model + ".zip")).exists()

    # string model + no project
    with pytest.raises(ValueError):
        get_zipped_model("mtest", None)

    # string model + project + no git repo
    tmp_dir = Path(tmp_path_factory.mktemp("test3"))
    model, project = get_zipped_model("mtest", tmp_dir, "ptest")
    assert ("mtest", "ptest") == (model, project)
    assert Path(tmp_dir / project / model / (model + ".zip")).exists()
