#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import shutil
from pathlib import Path

MODEL_PREFIX = "UNIT_TEST_MODEL"
MODEL = []


def test_pickle_trained_model():
    """
    Test cases:
    - normal
    - h2o binary
    - h2o mojo
    - binary string
    - mlflow model

    """
    from sasctl.pzmm.pickle_model import PickleModel as pm

    tmp_dir = tempfile.TemporaryDirectory()

    # Default model case
    pm.pickle_trained_model(
        trained_model=MODEL, model_prefix=MODEL_PREFIX, pickle_path=tmp_dir.name
    )
    assert (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).exists()
    (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).unlink()

    # H2O binary model case
    f = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir.name)
    shutil.copy(f.name, str(Path(tmp_dir.name) / MODEL_PREFIX))
    f.close()
    pm.pickle_trained_model(
        trained_model=None,
        model_prefix=MODEL_PREFIX,
        pickle_path=tmp_dir.name,
        is_h2o_model=True,
        is_binary_model=True,
    )
    assert (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).exists()
    (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).unlink()

    # H2O MOJO model case
    f = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir.name)
    pm.pickle_trained_model(
        trained_model=f.name,
        model_prefix=MODEL_PREFIX,
        pickle_path=tmp_dir.name,
        is_h2o_model=True,
    )
    assert (Path(tmp_dir.name) / (MODEL_PREFIX + ".mojo")).exists()
    (Path(tmp_dir.name) / (MODEL_PREFIX + ".mojo")).unlink()

    # Binary string case
    binary_string = pm.pickle_trained_model(
        trained_model=MODEL, model_prefix=MODEL_PREFIX, is_binary_string=True
    )
    assert isinstance(binary_string, str)

    # MLFlow model case
    mlflow_tmp_dir = tempfile.TemporaryDirectory()
    f = tempfile.NamedTemporaryFile(
        delete=False, dir=mlflow_tmp_dir.name, suffix=".pickle"
    )
    mlflow_dict = {"mlflowPath": mlflow_tmp_dir.name, "model_path": Path(f.name).name}
    # import pdb; pdb.set_trace()
    pm.pickle_trained_model(
        trained_model=None,
        model_prefix=MODEL_PREFIX,
        pickle_path=tmp_dir.name,
        mlflow_details=mlflow_dict,
    )
    assert (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).exists()
    (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).unlink()
