import pandas as pd
import pytest

onnx = pytest.importorskip("onnx")
torch = pytest.importorskip("torch")

import sasctl.utils.model_info
from sasctl.utils import get_model_info

# mnist
# get input/output shapes
# get var names if available
# classification/regression/etc
#

@pytest.fixture
def mnist_model(tmp_path):
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(14 * 14, 128)
            self.fc2 = torch.nn.Linear(128, 10)

        def forward(self, x):
            x = torch.nn.functional.max_pool2d(x, 2)
            x = x.reshape(-1, 1 * 14 * 14)
            x = self.fc1(x)
            x = torch.nn.functional.relu(x)
            x = self.fc2(x)
            output = torch.nn.functional.softmax(x, dim=1)
            return output

    model = Net()

    path = tmp_path / "model.onnx"
    X = torch.randn(1, 1, 28, 28)
    torch.onnx.export(model, X, path, input_names=["image"], output_names=["digit"])
    yield onnx.load(path), X


def test_get_info(mnist_model):
    info = get_model_info(*mnist_model)
    assert isinstance(info, sasctl.utils.model_info.OnnxModelInfo)

    # Output be classification into 10 digits
    assert len(info.output_column_names) == 10
    assert all(c.startswith("digit") for c in info.output_column_names)

    assert isinstance(info.X, pd.DataFrame)
    assert len(info.X.columns) == 28 * 28

    assert info.is_classifier
    assert not info.is_binary_classifier
    assert not info.is_regressor
    assert not info.is_clusterer