# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import json


class MLFlowModel:
    @classmethod
    def read_mlflow_model_file(cls, m_path=Path.cwd()):
        """
        Read and return model metadata and input/output variables as dictionaries from an MLFlow model directory.

        Current implementation only handles simple pickled models. Future feature work is required to include
        more types of MLFlow models.

        Parameters
        ----------
        m_path : str or Path object, optional
            Directory path of the MLFlow model files. Default is the current working directory.

        Returns
        -------
        var_dict : dict
            Model properties and metadata
        inputs_dict : list of dicts
            Model input variables
        outputs_dict : list of dicts
            Model output variables
        """
        with open(Path(m_path) / "MLmodel", "r") as m_file:
            m_lines = m_file.readlines()

        # Read in metadata and properties from the MLFlow model
        var_list = ["python_version", "serialization_format", "run_id", "model_path"]
        for i, var_string in enumerate(var_list):
            index = [i for i, s in enumerate(m_lines) if var_string in s]
            if not index:
                raise ValueError("This MLFlow model type is not currently supported.")
            var_list[i] = {var_list[i]: m_lines[index[0]].strip().split(" ")[1]}

        var_dict = {k: v for d in var_list for k, v in d.items()}
        var_dict["mlflowPath"] = m_path

        # Read in the input and output variables
        ind_in = [i for i, s in enumerate(m_lines) if "inputs:" in s]
        ind_out = [i for i, s in enumerate(m_lines) if "outputs:" in s]

        if ind_in and ind_out:
            inputs = m_lines[ind_in[0] : ind_out[0]]
            outputs = m_lines[ind_out[0] : -1]

            inputs_dict = json.loads("".join([s.strip() for s in inputs])[9:-1])
            outputs_dict = json.loads("".join([s.strip() for s in outputs])[10:-1])
        else:
            raise ValueError(
                "Improper or unset signature values for model. No input or output dicts could be generated."
            )
        return var_dict, inputs_dict, outputs_dict
