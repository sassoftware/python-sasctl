#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import tempfile
from pathlib import Path


def test_read_mlflow_model_file():
    """
    Test cases:
    - Improper file directory
    - Improper/unsupported MLmodel file
    - Unset signature values in model creation
    - Returns three dicts:
        - var_dict contains 5 values
        - inputs_dict is a list of dicts
        - outputs_dict is a list of dicts
    """
    from sasctl.pzmm.mlflow_model import MLFlowModel as ml

    tmp_dir = tempfile.TemporaryDirectory()

    # Improper file directory returns a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ml.read_mlflow_model_file(m_path=Path.cwd())

    # Invalid type of MLmodel file
    with open(Path(tmp_dir.name) / "MLmodel", "w") as f:
        f.write("Invalid MLmodel file.\nDoes not contain expected properties.")
    with pytest.raises(
        ValueError, match="This MLFlow model type is not currently supported."
    ):
        ml.read_mlflow_model_file(m_path=Path(tmp_dir.name))

    # Unset signature values in model creation
    with open(Path(tmp_dir.name) / "MLmodel", "w") as f:
        f.write(
            "artifact_path: model\nflavors:\n  python_function:\n    env: conda.yaml\n"
            "    loader_module: mlflow.sklearn\n    model_path: model.pkl\n    predict_fn: predict\n"
            "    python_version: 3.8.13\n  sklearn:\n    code: null\n    pickled_model: model.pkl\n"
            "    serialization_format: cloudpickle\n    sklearn_version: 1.1.2\nmlflow_version: 1.29.0"
            "\nmodel_uuid: 7e09461c218e48139c91971891f30888\nrun_id: ca7f64999a674c8c88b59eb36dcf2c75\n"
        )
    with pytest.raises(
        ValueError,
        match="Improper or unset signature values for model. No input or output dicts could be generated.",
    ):
        ml.read_mlflow_model_file(m_path=Path(tmp_dir.name))

    # Proper returns
    with open(Path(tmp_dir.name) / "MLmodel", "w") as f:
        f.write(
            "artifact_path: model\nflavors:\n  python_function:\n    env: conda.yaml\n    "
            "loader_module: mlflow.sklearn\n    model_path: model.pkl\n    predict_fn: predict\n    "
            "python_version: 3.8.13\n  sklearn:\n    code: null\n    pickled_model: model.pkl\n    "
            "serialization_format: cloudpickle\n    sklearn_version: 1.1.2\nmlflow_version: 1.29.0\n"
            "model_uuid: 7e09461c218e48139c91971891f30888\nrun_id: ca7f64999a674c8c88b59eb36dcf2c75\nsignature:\n"
            '  inputs: \'[{"name": "LOAN", "type": "long"}, {"name": "MORTDUE", "type": "double"},\n'
            '    {"name": "VALUE", "type": "double"}, {"name": "YOJ", "type": "double"}, {"name":\n'
            '    "DEROG", "type": "double"}, {"name": "DELINQ", "type": "double"}, {"name": "CLAGE",\n'
            '    "type": "double"}, {"name": "NINQ", "type": "double"}, {"name": "CLNO", "type":\n'
            '    "double"}, {"name": "DEBTINC", "type": "double"}, {"name": "JOB_Office", "type":\n'
            '    "integer"}, {"name": "JOB_Other", "type": "integer"}, {"name": "JOB_ProfExe",\n'
            '    "type": "integer"}, {"name": "JOB_Sales", "type": "integer"}, {"name": "JOB_Self",\n'
            '    "type": "integer"}, {"name": "REASON_HomeImp", "type": "integer"}]\'\n'
            '  outputs: \'[{"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1]}}]\'\n'
            "utc_time_created: '2022-10-14 18:08:51.098646'\n"
        )
    var_dict, inputs_dict, outputs_dict = ml.read_mlflow_model_file(
        m_path=Path(tmp_dir.name)
    )
    assert isinstance(var_dict, dict) and len(var_dict) == 5
    assert isinstance(inputs_dict, list) and isinstance(inputs_dict[0], dict)
    assert isinstance(outputs_dict, list) and isinstance(outputs_dict[0], dict)
