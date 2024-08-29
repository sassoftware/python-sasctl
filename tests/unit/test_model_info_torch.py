import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from sasctl.utils.model_info import get_model_info


def prepare_model_for_sas(model, model_name):
    import base64
    import inspect
    import pickle

    # Pickle the entire model
    pickled_model = pickle.dumps(model)

    # Save the state dict separately
    state_dict = model.state_dict()

    # Get the source code for the model class
    model_source = inspect.getsource(model.__class__)

    # Get the source for any custom modules used in the model
    custom_modules = {}
    for name, module in model.named_modules():
        if not hasattr(torch.nn, module.__class__.__name__):
            custom_modules[module.__class__.__name__] = inspect.getsource(
                module.__class__
            )

    # Capture initialization parameters
    init_signature = inspect.signature(model.__class__.__init__)
    init_params = {}
    for param_name, param in init_signature.parameters.items():
        if param_name == "self":
            continue
        if hasattr(model, param_name):
            init_params[param_name] = getattr(model, param_name)
        # else:
        #     # If the parameter is not stored as an attribute, we need to find its value
        #     # This is a bit tricky and might not work for all cases
        #     frame = inspect.currentframe()
        #     try:
        #         while frame:
        #             if param_name in frame.f_locals:
        #                 init_params[param_name] = frame.f_locals[param_name]
        #                 break
        #             frame = frame.f_back
        #     finally:
        #         del frame

    # Create a metadata dictionary
    metadata = {
        "model_name": model_name,
        "class_name": model.__class__.__name__,
        "pickled_model": base64.b64encode(pickle.dumps(model)).decode("utf-8"),
        "state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
        "model_source": inspect.getsource(model.__class__),
        "custom_modules": custom_modules,
        "init_params": init_params,
    }

    return metadata


class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias


def test_mnist():
    X = np.random.random(784).reshape(1, 784).astype("float32")

    model = MnistLogistic()
    info = get_model_info(model, X)

    meta = prepare_model_for_sas(model, "MnistLogistic")
    # assert info.is_classifier
