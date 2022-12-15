import pytest
import warnings
from sasctl.pzmm.modelParameters import ModelParameters as mp

pytestmark = pytest.mark.usefixtures("session")


@pytest.fixture
def train_data():
    """Returns the Iris data set as (X, y)"""

    try:
        import pandas as pd
    except ImportError:
        pytest.skip("Package `pandas` not found.")

    try:
        from sklearn import datasets
    except ImportError:
        pytest.skip("Package `sklearn` not found.")

    raw = datasets.load_iris()
    iris = pd.DataFrame(raw.data, columns=raw.feature_names)
    iris = iris.join(pd.DataFrame(raw.target))
    iris.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
    iris["Species"] = iris["Species"].astype("category")
    iris.Species.cat.categories = raw.target_names
    return iris.iloc[:, 0:4], iris["Species"]


@pytest.fixture
def sklearn_model(train_data):
    """Returns a simple Scikit-Learn model"""

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("Package `sklearn` not found.")

    X, y = train_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        model.fit(X, y)
    return model


@pytest.mark.incremental
class TestSklearnModel:
    PROJECT_NAME = "Test SKLearn Model"
    MODEL_NAME = "SKLearnModel"
    PATH = "."
