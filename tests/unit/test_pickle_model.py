#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
from pathlib import Path

import pytest

MODEL_PREFIX = "UNIT_TEST_MODEL"
MODEL = []


def test_pickle_trained_model():
    """
    Test cases:
    - normal
    - h2o binary (moved)
    - h2o mojo (moved)
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
    pm.pickle_trained_model(
        trained_model=None,
        model_prefix=MODEL_PREFIX,
        pickle_path=tmp_dir.name,
        mlflow_details=mlflow_dict,
    )
    assert (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).exists()
    (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).unlink()


def test_pickle_trained_model_h2o():
    """
    Side function for h2o models in case h2o is not installed.
    """
    h2o = pytest.importorskip("h2o")
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator

    from sasctl.pzmm.pickle_model import PickleModel as pm

    h2o.init()
    data = h2o.import_file("examples/data/hmeq.csv")
    data["BAD"] = data["BAD"].asfactor()
    y = "BAD"
    x = list(data.columns)
    x.remove(y)

    model = H2OGeneralizedLinearEstimator(
        family="binomial", model_id="test_model", lambda_search=True
    )
    model.train(x=x, y=y, training_frame=data)

    tmp_dir = tempfile.TemporaryDirectory()
    # H2O binary model case
    pm.pickle_trained_model(
        trained_model=model,
        model_prefix=MODEL_PREFIX,
        pickle_path=tmp_dir.name,
        is_h2o_model=True,
        is_binary_model=True,
    )
    assert (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).exists()
    model = h2o.load_model(tmp_dir.name + "/" + MODEL_PREFIX + ".pickle")
    (Path(tmp_dir.name) / (MODEL_PREFIX + ".pickle")).unlink()

    # H2O MOJO model case
    pm.pickle_trained_model(
        trained_model=model,
        model_prefix=MODEL_PREFIX,
        pickle_path=tmp_dir.name,
        is_h2o_model=True,
    )
    assert (Path(tmp_dir.name) / (MODEL_PREFIX + ".mojo")).exists()
    model = h2o.import_mojo(tmp_dir.name + "/" + MODEL_PREFIX + ".mojo")
    (Path(tmp_dir.name) / (MODEL_PREFIX + ".mojo")).unlink()
