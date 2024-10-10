# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import yaml
import json
from pathlib import Path


class MLFlowModel:
    @classmethod
    def read_mlflow_model_file(cls, m_path=Path.cwd()):
        """
        Read and return model metadata and input/output variables as dictionaries from
        an MLFlow model directory.

        Current implementation only handles simple pickled models. Future feature work
        is required to include more types of MLFlow models.

        Parameters
        ----------
        m_path : str or pathlib.Path, optional
            Directory path of the MLFlow model files. Default is the current working
            directory.

        Returns
        -------
        var_dict : dict
            Model properties and metadata
        inputs_dict : list of dict
            Model input variables
        outputs_dict : list of dict
            Model output variables
        """
        with open(Path(m_path) / "MLmodel", "r") as m_file:
            m_yml = yaml.safe_load(m_file)

        # Read in metadata and properties from the MLFlow model
        try:
            var_dict = {
                "python_version": m_yml["flavors"]["python_function"]["python_version"],
                "model_path": m_yml["flavors"]["python_function"]["model_path"],
                "serialization_format": m_yml["flavors"]["sklearn"][
                    "serialization_format"
                ],
                "run_id": m_yml["run_id"],
                "mlflowPath": m_path,
            }
        except KeyError:
            raise ValueError(
                "This MLFlow model type is not currently supported."
            ) from None
        except TypeError:
            raise ValueError(
                "This MLFlow model type is not currently supported."
            ) from None

        # Read in the input and output variables
        try:
            inputs_dict = json.loads(m_yml["signature"]["inputs"])
            outputs_dict = json.loads(m_yml["signature"]["outputs"])
        except KeyError:
            raise ValueError(
                "Improper or unset signature values for model. No input or output "
                "dicts could be generated. "
            ) from None

        return var_dict, inputs_dict, outputs_dict
