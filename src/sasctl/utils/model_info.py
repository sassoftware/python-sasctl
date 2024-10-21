#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

try:
    import onnx

    # ONNX serializes models using protobuf, so this should be safe
    from google.protobuf import json_format
except ImportError:
    onnx = None

try:
    import onnxruntime
except ImportError:
    onnxruntime = None


def get_model_info(model, X, y=None):
    """Extracts metadata about the model and associated data sets.

    Parameters
    ----------
    model : object
        A trained model
    X : array-like
        Sample of the data used to train the model.
    y : array-like
        Sample of the output produced by the model.

    Returns
    -------
    ModelInfo

    Raises
    ------
    ValueError
        If `model` is not a recognized type.

    """

    # Don't need to import sklearn, just check if the class is part of that module.
    if type(model).__module__.startswith("sklearn."):
        return SklearnModelInfo(model, X, y)

    if type(model).__module__.startswith("onnx"):
        return _load_onnx_model(model, X, y)

    # Most PyTorch models are actually subclasses of torch.nn.Module, so checking module
    # name alone is not sufficient.
    if torch and isinstance(model, torch.nn.Module):
        return PyTorchModelInfo(model, X, y)

    raise ValueError(f"Unrecognized model type {type(model)} received.")


def _load_onnx_model(model, X, y=None):
    # TODO: unncessary?  static analysis of onnx file sufficient?
    if onnxruntime:
        return OnnxModelInfo(model, X, y)

    return OnnxModelInfo(model, X, y)


class ModelInfo(ABC):
    """Base class for storing model metadata.

    Attributes
    ----------
    algorithm : str
        Will appear in the "Algorithm" drop-down menu in Model Manager.
        Example: "Forest", "Neural networks", "Binning", etc.
    analytic_function : str
        Will appear in the "Function" drop-down menu in Model Manager.
        Example: "Classification", "Clustering", "Prediction"
    is_binary_classifier : bool
    is_classifier : bool
    is_regressor : bool
    is_clusterer : bool
    model : object
        The model instance that the information was extracted from.
    model_params : {str: any}
        Dictionary of parameter names and values.
    output_column_names : list of str
        Variable names associated with the outputs of `model`.
    predict_function : callable
        The method on `model` that is called to produce predicted values.
    target_values : list of str
        Class labels returned by a classification model.  For binary classification models
        this is just the label of the targeted event level.
    threshold : float or None
        The cutoff value used in a binary classification model to determine which class an
        observation belongs to.  Returns None if not a binary classification model.
    X : pandas.DataFrame
        A sample of the input data used to train the model.  Must contain at least one row.
    y : pandas.DataFrame
        A sample of the output data produced by the model.  Must have at least one row.

    """

    @property
    @abstractmethod
    def algorithm(self) -> str:
        return

    @property
    def analytic_function(self) -> str:
        if self.is_classifier:
            return "classification"
        if self.is_regressor:
            return "prediction"

    @property
    def description(self) -> str:
        return str(self.model)

    @property
    @abstractmethod
    def is_binary_classifier(self) -> bool:
        return

    @property
    @abstractmethod
    def is_classifier(self) -> bool:
        return

    @property
    @abstractmethod
    def is_clusterer(self) -> bool:
        return

    @property
    @abstractmethod
    def is_regressor(self) -> bool:
        return

    @property
    @abstractmethod
    def model(self) -> object:
        return

    @property
    @abstractmethod
    def model_params(self) -> Dict[str, Any]:
        return

    @property
    def output_column_names(self) -> List[str]:
        return self.y.columns.tolist()

    @property
    @abstractmethod
    def predict_function(self) -> Callable:
        return

    @property
    @abstractmethod
    def target_column(self):
        return

    @property
    @abstractmethod
    def target_values(self):
        # "target event"
        # value that indicates the target event has occurred in bianry classification
        return

    @property
    @abstractmethod
    def threshold(self) -> Union[str, None]:
        return

    @property
    @abstractmethod
    def X(self) -> pd.DataFrame:
        return

    @property
    @abstractmethod
    def y(self) -> pd.DataFrame:
        return


class OnnxModelInfo(ModelInfo):
    def __init__(self, model, X, y=None):
        if onnx is None:
            raise RuntimeError(
                "The onnx package must be installed to work with ONNX models.  "
                "Please `pip install onnx`."
            )

        self._model = model
        self._X = X
        self._y = y

        inferred_model = onnx.shape_inference.infer_shapes(model)

        inputs = [self._tensor_to_dataframe(i) for i in inferred_model.graph.input]
        outputs = [self._tensor_to_dataframe(o) for o in inferred_model.graph.output]

        if len(inputs) > 1:
            warnings.warn(
                f"The ONNX model has {len(inputs)} inputs but only the first input "
                f"will be captured in Model Manager."
            )

        if len(outputs) > 1:
            warnings.warn(
                f"The ONNX model has {len(outputs)} outputs but only the first output "
                f"will be captured in Model Manager."
            )

        self._X_df = inputs[0]
        self._y_df = outputs[0]

    @staticmethod
    def _tensor_to_dataframe(tensor):
        """

        Parameters
        ----------
        tensor : onnx.onnx_ml_pb2.ValueInfoProto or dict
            A protobuf `Message` containing information

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        df = _tensor_to_dataframe(model.graph.input[0])

        """
        if isinstance(tensor, onnx.onnx_ml_pb2.ValueInfoProto):
            tensor = json_format.MessageToDict(tensor)
        elif not isinstance(tensor, dict):
            raise ValueError(f"Unexpected type {type(tensor)}.")

        name = tensor.get("name", "Var")
        type_ = tensor["type"]

        if "tensorType" not in type_:
            raise ValueError(f"Received an unexpected ONNX input type: {type_}.")

        dtype = onnx.helper.tensor_dtype_to_np_dtype(type_["tensorType"]["elemType"])

        # Tuple of tensor dimensions e.g. (1, 1, 24)
        input_dims = tuple(
            int(d["dimValue"]) for d in type_["tensorType"]["shape"]["dim"]
        )

        return pd.DataFrame(
            dtype=dtype, columns=[f"{name}{i+1}" for i in range(math.prod(input_dims))]
        )

    @property
    def algorithm(self) -> str:
        return "neural network"

    @property
    def description(self) -> str:
        return self.model.doc_string

    @property
    def is_binary_classifier(self) -> bool:
        return len(self.output_column_names) == 2

    @property
    def is_classifier(self) -> bool:
        return len(self.output_column_names) > 1

    @property
    def is_clusterer(self) -> bool:
        return False

    @property
    def is_regressor(self) -> bool:
        return len(self.output_column_names) == 1

    @property
    def model(self) -> object:
        return self._model

    @property
    def model_params(self) -> Dict[str, Any]:
        return {
            k: getattr(self.model, k, None)
            for k in (
                "ir_version",
                "model_version",
                "opset_import",
                "producer_name",
                "producer_version",
            )
        }

    @property
    def predict_function(self) -> Callable:
        return None

    @property
    def target_column(self):
        return None

    @property
    def target_values(self):
        return None

    @property
    def threshold(self) -> Union[str, None]:
        return None

    @property
    def X(self) -> pd.DataFrame:
        return self._X_df

    @property
    def y(self) -> pd.DataFrame:
        return self._y_df


class PyTorchModelInfo(ModelInfo):
    """Stores model information for a PyTorch model instance."""

    def __init__(self, model, X, y=None):
        if torch is None:
            raise RuntimeError(
                "The PyTorch package must be installed to work with PyTorch models.  Please `pip install torch`."
            )

        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected PyTorch model, received {type(model)}.")

        # Some models may take multiple tensors as input.  These can be passed as a tuple
        # of tensors.  To simplify processing, convert even single inputs into tuples.
        if not isinstance(X, tuple):
            X = (X,)

        for x in X:
            if not isinstance(x, (np.ndarray, torch.Tensor)):
                raise ValueError(
                    f"Expected input data to be a numpy array or PyTorch tensor, received {type(X)}."
                )

        # Ensure each input is a PyTorch Tensor
        X = tuple(x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in X)

        # Store the current setting so that we can restore it later
        is_training = model.training

        if y is None:
            model.eval()

            with torch.no_grad():
                y = model(*X)

        if not isinstance(y, (np.ndarray, torch.Tensor)):
            raise ValueError(
                f"Expected output data to be a numpy array or PyTorch tensor, received {type(y)}."
            )

        self._model = model
        self._X = X
        self._y = y

        # Model Manager doesn't currently support arrays or vectors.  Capture the first
        # input tensor and reshape to 2 dimensions if necessary.
        x0 = X[0]
        if x0.ndim > 2:
            x0 = x0.reshape((x0.shape[0], -1))
        self._X_df = pd.DataFrame(x0, columns=[f"Var{i+1}" for i in range(x0.shape[1])])

        # Flatten to 2 dimensions if necessary
        if y.ndim > 2:
            y = y.reshape((y.shape[0], -1))

        self._y_df = pd.DataFrame(y, columns=[f"Out{i+1}" for i in range(y.shape[1])])

        self._layer_info = self._get_layer_info(model, X)

        # Reset the model to its original training state
        model.train(is_training)

    @staticmethod
    def _get_layer_info(model, X):
        """Run data through the model to determine layer types and tensor shapes.

        Parameters
        ----------
        model : torch.nn.Module
        X : torch.Tensor

        Returns
        -------
        List[Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]]

        """
        is_training = model.training

        # Track the registered hooks so we can unregister them after running.
        # NOTE: if not removed, hooks can prevent model from being sucessfully pickled.
        hooks = []

        # Track the layers and their input/output tensors for later reference.
        layers = []

        def hook(module, input, output, *args):
            layers.append((module, input, output))

        for module in model.modules():
            handle = module.register_forward_hook(hook)
            hooks.append(handle)

        model.eval()
        with torch.no_grad():
            model(*X)

        for handle in hooks:
            handle.remove()

        model.train(is_training)
        return layers

    @property
    def algorithm(self):
        return "PyTorch"

    @property
    def is_binary_classifier(self):
        return False

    @property
    def is_classifier(self):
        return False

    @property
    def is_clusterer(self):
        return False

    @property
    def is_regressor(self):
        return False

    @property
    def model(self):
        return self._model

    @property
    def model_params(self) -> Dict[str, Any]:
        return self.model.__dict__

    @property
    def output_column_names(self):
        return list(self.y.columns)

    @property
    def predict_function(self):
        return self.model.forward

    @property
    def target_column(self):
        return self.y.columns[0]

    @property
    def target_values(self):
        return []

    @property
    def threshold(self):
        return None

    @property
    def X(self):
        return self._X_df

    @property
    def y(self):
        return self._y_df


class SklearnModelInfo(ModelInfo):
    """Stores model information for a scikit-learn model instance."""

    # Map class names from sklearn to algorithm names used by SAS
    _algorithm_mappings = {
        "LogisticRegression": "Logistic regression",
        "LinearRegression": "Linear regression",
        "SVC": "Support vector machine",
        "SVR": "Support vector machine",
        "GradientBoostingClassifier": "Gradient boosting",
        "GradientBoostingRegressor": "Gradient boosting",
        "RandomForestClassifier": "Forest",
        "RandomForestRegressor": "Forest",
        "DecisionTreeClassifier": "Decision tree",
        "DecisionTreeRegressor": "Decision tree",
    }

    def __init__(self, model, X, y):
        is_classifier = hasattr(model, "classes_")
        is_binary_classifier = is_classifier and len(model.classes_) == 2
        is_clusterer = hasattr(model, "cluster_centers_")

        if y is None:
            if hasattr(model, "predict_proba"):
                y = model.predict_proba(X)
            else:
                y = model.predict(X)

        # Ensure input/output is a DataFrame for consistency
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)

        # If not a classfier or a clustering algorithm and output is a single column, then
        # assume its a regression algorithm
        is_regressor = (
            not is_classifier
            and not is_clusterer
            and (y_df.shape[1] == 1 or "Regress" in type(model).__name__)
        )

        if not is_classifier and not is_regressor and not is_clusterer:
            raise ValueError(f"Unexpected model type {model} received.")

        self._is_classifier = is_classifier
        self._is_binary_classifier = is_binary_classifier
        self._is_regressor = is_regressor
        self._is_clusterer = is_clusterer
        self._model = model

        if not hasattr(y, "name") and not hasattr(y, "columns"):
            # If example output doesn't contain column names then our DataFrame equivalent
            # also lacks good column names.  Assign reasonable names for use downstream.
            if y_df.shape[1] == 1:
                y_df.columns = ["I_Target"]
            elif self.is_classifier:
                # Output is probability of each label.  Name columns according to classes.
                y_df.columns = [f"P_{class_}" for class_ in model.classes_]
            elif not y_df.empty:
                # If we were passed data for `y` but we don't know the format raise an error.
                # This *shouldn't* happen unless a cluster algorithm somehow produces wide output.
                raise ValueError(f"Unrecognized model output format.")

        # Store the data sets for reference later.
        self._X = X_df
        self._y = y_df

    @property
    def algorithm(self):
        # Get the model or the last step in the Pipeline
        estimator = getattr(self.model, "_final_estimator", self.model)
        estimator = type(estimator).__name__

        # Convert the class name to an algorithm, or return the class name if no match.
        return self._algorithm_mappings.get(estimator, estimator)

    @property
    def is_binary_classifier(self):
        return self._is_binary_classifier

    @property
    def is_classifier(self):
        return self._is_classifier

    @property
    def is_clusterer(self):
        return self._is_clusterer

    @property
    def is_regressor(self):
        return self._is_regressor

    @property
    def model(self):
        return self._model

    @property
    def model_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    @property
    def output_column_names(self):
        return list(self._y.columns)

    @property
    def predict_function(self):
        # If desired output has multiple columns then we can assume its the probability values
        if self._y.shape[1] > 1 and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba

        # Otherwise its the single value from .predict()
        return self.model.predict

    @property
    def target_column(self):
        return self.y.columns[0]

    @property
    def target_values(self):
        if self.is_binary_classifier:
            return [self.model.classes_[-1]]
        if self.is_classifier:
            return list(self.model.classes_)

    @property
    def threshold(self):
        # sklearn seems to always use 0.5 as a cutoff for .predict()
        if self.is_binary_classifier:
            return 0.5

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y
