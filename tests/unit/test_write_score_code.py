#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sasctl.pzmm as pzmm
from sasctl import current_session
from sasctl.core import RestObj, VersionInfo
from sasctl.pzmm.write_score_code import ScoreCode as sc
from sasctl.pzmm.write_score_code import ScoreCode


@pytest.fixture()
def predict_proba():
    pass


@pytest.fixture()
def score_code_mocks():
    with patch.multiple(
        "sasctl.pzmm.ScoreCode",
        _input_var_lists=DEFAULT,
        _check_viya_version=DEFAULT,
        _write_imports=MagicMock(),
        _viya35_model_load=DEFAULT,
        _viya4_model_load=DEFAULT,
        _check_valid_model_prefix=DEFAULT,
        _impute_missing_values=MagicMock(),
        _predict_method=MagicMock(),
        _predictions_to_metrics=MagicMock(),
        _viya35_score_code_import=DEFAULT,
    ) as mocks:
        yield mocks


def test_get_model_id():
    """
    Test Cases:
    - No model provided
    - Model not found
    - Model found
    """
    with pytest.raises(ValueError):
        sc._get_model_id(None)

    with patch("sasctl._services.model_repository.ModelRepository.get_model") as model:
        model.return_value = None

        with pytest.raises(ValueError):
            sc._get_model_id("DNEModel")

        model.return_value = RestObj(name="test_model", id="123456")
        assert sc._get_model_id("test_model") == "123456"


def test_check_for_invalid_variable_names():
    """
    Test Cases:
    - Invalid variable(s) found
    - Valid variables only
    """
    var_list = ["bad_variable", "good_variable", "awful variable"]
    with pytest.raises(SyntaxError):
        sc._check_for_invalid_variable_names(var_list)
    try:
        var_list = ["good_variable", "best_variable"]
        sc._check_for_invalid_variable_names(var_list)
    except SyntaxError:
        pytest.fail("test_check_for_invalid_variable_names improperly raised an error")


def test_write_imports():
    """
    Test Cases:
    - Viya 3.5 connection
    - pickle_type is anything else
    - Viya 4 connection
    - pickle_type is "pickle"
    - h2o model (mojo or binary)
    - binary string model
    """
    with patch("sasctl.core.Session._get_authorization_token"):
        current_session("example.com", "user", "password")

    with patch("sasctl.core.Session.version_info") as version:
        version.return_value = VersionInfo(3)
        sc._write_imports(pickle_type="dill")
        assert "import settings" not in sc.score_code
        assert "import dill" in sc.score_code
        sc.score_code = ""

        version.return_value = VersionInfo(4)
        sc._write_imports(mojo_model=True)
        assert "import settings" in sc.score_code
        assert "import h2o" in sc.score_code
        sc.score_code = ""

        sc._write_imports(binary_string=b"test binary string")
        assert "import codecs" in sc.score_code
        sc.score_code = ""


def test_viya35_model_load():
    """
    Test Cases:
    - non-h2o model
    - mojo model
    - binary h2o model
    """
    load_text = sc._viya35_model_load("1234", "normal")
    assert "pickle.load(pickle_model)" in sc.score_code
    assert "pickle.load(pickle_model)" in load_text
    sc.score_code = ""

    mojo_text = sc._viya35_model_load("2345", "mojo", mojo_model=True)
    assert "h2o.import_mojo" in sc.score_code
    assert "h2o.import_mojo" in mojo_text
    sc.score_code = ""

    binary_text = sc._viya35_model_load("3456", "binary", binary_h2o_model=True)
    assert "h2o.load" in sc.score_code
    assert "h2o.load" in binary_text
    sc.score_code = ""


def test_viya4_model_load():
    """
    Test Cases:
    - non-h2o model
    - mojo model
    - binary h2o model
    - Tensorflow keras model
    """
    load_text = sc._viya4_model_load("normal")
    assert "pickle.load(pickle_model)" in sc.score_code
    assert "pickle.load(pickle_model)" in load_text
    sc.score_code = ""

    mojo_text = sc._viya4_model_load("mojo", mojo_model=True)
    assert "h2o.import_mojo" in sc.score_code
    assert "h2o.import_mojo" in mojo_text
    sc.score_code = ""

    binary_text = sc._viya4_model_load("binary", binary_h2o_model=True)
    assert "h2o.load" in sc.score_code
    assert "h2o.load" in binary_text
    sc.score_code = ""

    keras_text = sc._viya4_model_load("tensorflow", tf_keras_model=True)
    assert "tf.keras.models.load_model" in sc.score_code
    assert "tf.keras.models.load_model" in keras_text


def test_impute_missing_values():
    """
    Test Cases:
    - numeric data
    - character data
    """
    test_df = pd.DataFrame(
        data=[[0, "a", 1], [2, "b", 0]], columns=["num", "char", "bin"]
    )
    sc._impute_missing_values(test_df, True)
    assert "'num': 1" in sc.score_code
    assert "'char': ''" in sc.score_code
    assert "'bin': 0" in sc.score_code

    sc._impute_missing_values(test_df, [5, "test", 1])
    assert "'num': 5" in sc.score_code
    assert "'char': 'test'" in sc.score_code
    assert "'bin': 1" in sc.score_code

    sc._impute_missing_values(test_df, {"a": 5, "b": "test", "c": 1})
    assert "'a': 5" in sc.score_code
    assert "'b': 'test'" in sc.score_code
    assert "'c': 1" in sc.score_code


def test_predict_method():
    """
    Test Cases:
    - normal model
    - h2o model, based of dtype_list input
    - statsmodels model
    """
    var_list = ["first", "second", "third"]
    dtype_list = ["str", "int", "float"]
    sc._predict_method(predict_proba, var_list)
    assert '{"first": first, "second": second' in sc.score_code
    sc.score_code = ""

    sc._predict_method(predict_proba, var_list, dtype_list=dtype_list)
    assert "column_types = " in sc.score_code
    sc.score_code = ""

    sc._predict_method(predict_proba, var_list, statsmodels_model=True)
    assert '{"const": const, "first": first' in sc.score_code
    sc.score_code = ""


def test_determine_returns_type():
    """
    Test cases:
    - example values
    - example types
    - mixture of values/types
    - unexpected types
    """
    assert sc._determine_returns_type(["TestReturn", 1.2]) == [True, False]
    assert sc._determine_returns_type([int, float, str]) == [False, False, True]
    assert sc._determine_returns_type(["TestReturn", int, str]) == [True, False, True]
    assert sc._determine_returns_type([dict]) == [True]


def test_yield_score_metrics():
    """
    Test cases:
    - Any order for classification/prediction values
    - No target variable provided
    """
    metrics = sc._yield_score_metrics(
        [False, True, False], ["Math", "English"], "ClassyVar"
    )
    assert [x for x in metrics] == ["P_Math", "I_ClassyVar", "P_English"]

    with pytest.warns():
        metrics = sc._yield_score_metrics([False, True, False], ["Math", "English"])
        assert [x for x in metrics] == ["P_Math", "I_Classification", "P_English"]


def test_determine_score_metrics():
    """
    Test cases:
    - matrix...
        - len(target_values) = [0, 1, 2, >2]
        - len(predict_returns == True) = [0, 1, >1]
        - len(predict_returns == False) = [0, =len(tv), Any]
        - target_variable = [None, Any]
    """
    assert sc._determine_score_metrics([float], "TestPredict", None) == [
        f"I_TestPredict"
    ]

    with pytest.warns():
        assert sc._determine_score_metrics([float], None, None) == [f"I_Prediction"]

    with pytest.raises(ValueError):
        sc._determine_score_metrics([float, int], None, None)

    with pytest.raises(ValueError):
        sc._determine_score_metrics([float], None, ["A"])

    with pytest.raises(ValueError):
        sc._determine_score_metrics([float], None, "A")

    with pytest.raises(ValueError):
        sc._determine_score_metrics([float, float, int], None, ["A", "B"])

    with pytest.raises(ValueError):
        sc._determine_score_metrics([str, str], None, ["A", "B"])

    assert sc._determine_score_metrics([float], None, ["A", "B"]) == ["P_A"]
    assert sc._determine_score_metrics([float, int], None, ["A", "B"]) == ["P_A", "P_B"]
    assert sc._determine_score_metrics([str], None, ["A", "B"]) == ["I_Classification"]
    assert sc._determine_score_metrics([str, float], None, ["A", "B"]) == [
        "I_Classification",
        "P_A",
    ]
    assert sc._determine_score_metrics([float, str, int], "Classy", ["A", "B"]) == [
        "P_A",
        "I_Classy",
        "P_B",
    ]

    with pytest.raises(ValueError):
        sc._determine_score_metrics([str, str], None, ["A", "B", "C"])

    with pytest.raises(ValueError):
        sc._determine_score_metrics([str, float, float], None, ["A", "B", "C"])

    assert sc._determine_score_metrics([str], None, ["A", "B", "C"]) == [
        "I_Classification"
    ]
    assert sc._determine_score_metrics(
        [str, float, float, float], None, ["A", "B", "C"]
    ) == ["I_Classification", "P_A", "P_B", "P_C"]
    assert sc._determine_score_metrics(
        [float, float, float], None, ["A", "B", "C"]
    ) == ["P_A", "P_B", "P_C"]


class TestNoTargetsNoThresholds(unittest.TestCase):
    def setUp(self):
        self.sc = ScoreCode

    def tearDown(self):
        self.sc.score_code = ""

    def execute_snippet(self, *args):
        scope = {}
        exec(self.sc.score_code, scope)
        test_snippet = scope["test_snippet"]
        return test_snippet(*args)

    def test_improper_arguments(self):
        metrics = "Classification"
        returns = [1, "A"]
        with pytest.raises(ValueError):
            self.sc._no_targets_no_thresholds(metrics, returns)

    def test_single_metric(self):
        metrics = "Classification"
        returns = [1]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._no_targets_no_thresholds(metrics, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0.5]]
        self.assertEqual(self.execute_snippet(input_array, prediction), 0.5)
        # Multi row
        input_array = pd.DataFrame({"A": [0.9, 1, 1.1]})
        prediction = [0.3, 0.4, 0.5]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({metrics: prediction}),
        )

    def test_single_metric_h2o(self):
        metrics = "Classification"
        returns = [1]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._no_targets_no_thresholds(metrics, returns, h2o_model=True)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[], [0.5]]
        self.assertEqual(self.execute_snippet(input_array, prediction), 0.5)
        # Multi row
        input_array = pd.DataFrame({"A": [0.9, 1, 1.1]})
        prediction = pd.DataFrame({"predict": [0, 1, 1], "p0": [0.3, 0.4, 0.5]})
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": [0, 1, 1]}),
        )

    def test_multi_metric(self):
        metrics = ["Classification", "Proba_A", "Proba_B", "Proba_C"]
        returns = ["I", 1, 2, 3]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._no_targets_no_thresholds(metrics, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [["i", 0.3, 0.4, 0.5]]
        self.assertEqual(
            self.execute_snippet(input_array, prediction), ("i", 0.3, 0.4, 0.5)
        )
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = pd.DataFrame(
            {
                "Classification": ["i", "j", "k"],
                "Proba_A": [0.1, 0.2, 0.3],
                "Proba_B": [0.4, 0.5, 0.6],
                "Proba_C": [0.7, 0.8, 0.9],
            }
        )
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction), prediction
        )

    def test_multi_metric_h2o(self):
        metrics = ["Classification", "Proba_A", "Proba_B", "Proba_C"]
        returns = ["I", 1, 2, 3]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._no_targets_no_thresholds(metrics, returns, h2o_model=True)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[], ["i", 0.3, 0.4, 0.5]]
        self.assertEqual(
            self.execute_snippet(input_array, prediction), ("i", 0.3, 0.4, 0.5)
        )
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = pd.DataFrame(
            {
                "Classification": ["i", "j", "k"],
                "Proba_A": [0.1, 0.2, 0.3],
                "Proba_B": [0.4, 0.5, 0.6],
                "Proba_C": [0.7, 0.8, 0.9],
            }
        )
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction), prediction
        )


class TestBinaryTarget(unittest.TestCase):
    def setUp(self):
        self.sc = ScoreCode
        self.target_values = ["A", "B"]

    def tearDown(self):
        self.sc.score_code = ""

    def execute_snippet(self, *args):
        scope = {}
        exec(self.sc.score_code, scope)
        test_snippet = scope["test_snippet"]
        return test_snippet(*args)

    def test_improper_arguments(self):
        with pytest.raises(ValueError):
            sc._binary_target([], [], ["A", 1, 2, 3])
        with pytest.raises(ValueError):
            sc._binary_target([], [], ["A", "B"])
        with pytest.raises(ValueError):
            sc._binary_target(["A", "B", "C", "D"], [], [])

    def test_one_metric_h2o(self):
        metrics = "Classification"
        returns = [1, 0, 0]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns, h2o_model=True)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[], [1, 2, 3]]
        self.assertEqual(self.execute_snippet(input_array, prediction), "B")
        # Multi row
        input_array = pd.DataFrame({"A": [0, 1]})
        prediction = pd.DataFrame(
            {"predict": [0, 1], "p0": [0.9, 0.2], "p1": [0.1, 0.8]}
        )
        output_table = pd.DataFrame({"Classification": ["A", "B"]})
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction), output_table
        )

    def test_one_metric_one_return_classification(self):
        metrics = "Classification"
        returns = [""]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [0.5]
        self.assertEqual(self.execute_snippet(input_array, prediction), 0.5)
        # Multi row
        input_array = pd.DataFrame({"A": [0.9, 1, 1.1]})
        prediction = [0.3, 0.4, 0.5]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({metrics: prediction}),
        )

    def test_one_metric_one_return_probability(self):
        metrics = "Classification"
        returns = [int]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [1]
        self.assertEqual(self.execute_snippet(input_array, prediction), "B")
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [0, 1, 0]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({metrics: ["A", "B", "A"]}),
        )

    def test_one_metric_two_returns(self):
        metrics = "Classification"
        returns = [int, int]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[1, 0]]
        self.assertEqual(self.execute_snippet(input_array, prediction), "A")
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0, 1], [1, 0]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({metrics: ["B", "A"]}),
        )

    def test_one_metric_three_returns(self):
        metrics = "Classification"
        returns = [int, "", int]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0, "Y", "z"]]
        self.assertEqual(self.execute_snippet(input_array, prediction), "Y")
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0, "A", "b"], ["1", "B", 0.4]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({metrics: ["A", "B"]}),
        )

    def test_two_metrics_h2o(self):
        metrics = ["Classification", "Probability"]
        returns = ["", int, int]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns, h2o_model=True)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[], ["a", -1, 1]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("B", 1.0))
        # Multi row
        input_array = pd.DataFrame({"A": [0, 1]})
        prediction = pd.DataFrame(
            {"predict": [0, 1], "p0": [0.9, 0.2], "p1": [0.1, 0.8]}
        )
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "B"], "Probability": [0.1, 0.8]}),
        )

    def test_two_metrics_one_return(self):
        metrics = ["Classification", "Probability"]
        returns = [int]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [0.2]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("A", 0.2))
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [1, -1]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["B", "A"], "Probability": [1, -1]}),
        )

    def test_two_metrics_two_returns_no_classification(self):
        metrics = ["Classification", "Probability"]
        returns = [int, int]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0.2, 0.8]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("B", 0.8))
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0.9, 0.1], [0.4, 0.6]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "B"], "Probability": [0.1, 0.6]}),
        )

    def test_two_metrics_two_returns_classification(self):
        metrics = ["Classification", "Probability"]
        returns = ["", int]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [["B", 0.2]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("B", 0.2))
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [["A", 0.9], ["B", 0.4]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "B"], "Probability": [0.9, 0.4]}),
        )

    def test_two_metrics_three_returns_class_first(self):
        metrics = ["Classification", "Probability"]
        returns = ["", int, int]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [["B", 0.2, 0.8]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("B", 0.8))
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [["A", 0.9, 0.1], ["B", 0.4, 0.6]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "B"], "Probability": [0.1, 0.6]}),
        )

    def test_two_metrics_three_returns_class_last(self):
        metrics = ["Classification", "Probability"]
        returns = [int, int, ""]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0.2, 0.8, "B"]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("B", 0.8))
        # Multi row
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0.9, 0.1, "A"], [0.4, 0.6, "B"]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "B"], "Probability": [0.1, 0.6]}),
        )

    def test_three_metrics_h2o(self):
        metrics = ["Classification", "Proba_0", "Proba_1"]
        returns = ["", int, int]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns, h2o_model=True)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[], ["a", -1, 1]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("a", -1, 1))
        # Multi row
        input_array = pd.DataFrame({"A": [0, 1]})
        prediction = pd.DataFrame(
            {"predict": [0, 1], "p0": [0.9, 0.2], "p1": [0.1, 0.8]}
        )
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {"Classification": [0, 1], "Proba_0": [0.9, 0.2], "Proba_1": [0.1, 0.8]}
            ),
        )

    def test_three_metrics_one_return(self):
        metrics = ["Classification", "Proba_0", "Proba_1"]
        returns = [int]
        self.sc.score_code += (
            "import pandas as pd\n" "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        print(self.sc.score_code)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [0.9]
        self.assertEqual(
            self.execute_snippet(input_array, prediction), ("B", 0.9, 1 - 0.9)
        )
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [0.9, 0.1]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {
                    "Classification": ["B", "A"],
                    "Proba_0": [0.9, 0.1],
                    "Proba_1": [1 - 0.9, 1 - 0.1],
                }
            ),
        )

    def test_three_metrics_two_returns_no_class(self):
        metrics = ["Classification", "Proba_0", "Proba_1"]
        returns = [int, int]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0.9, 0.1]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("A", 0.1, 0.9))
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0.9, 0.1], [0.2, 0.8]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {
                    "Classification": ["A", "B"],
                    "Proba_0": [0.9, 0.2],
                    "Proba_1": [0.1, 0.8],
                }
            ),
        )

    def test_three_metrics_two_returns_class_first(self):
        metrics = ["Classification", "Proba_0", "Proba_1"]
        returns = ["", int]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [["A", 0.9]]
        self.assertEqual(
            self.execute_snippet(input_array, prediction), ("A", 0.9, 1 - 0.9)
        )
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [["A", 0.9], ["B", 0.2]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {
                    "Classification": ["A", "B"],
                    "Proba_0": [0.9, 0.2],
                    "Proba_1": [1 - 0.9, 1 - 0.2],
                }
            ),
        )

    def test_three_metrics_two_returns_class_last(self):
        metrics = ["Classification", "Proba_0", "Proba_1"]
        returns = [int, ""]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0.9, "A"]]
        self.assertEqual(
            self.execute_snippet(input_array, prediction), ("A", 0.9, 1 - 0.9)
        )
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0.9, "A"], [0.2, "B"]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {
                    "Classification": ["A", "B"],
                    "Proba_0": [0.9, 0.2],
                    "Proba_1": [1 - 0.9, 1 - 0.2],
                }
            ),
        )

    def test_three_metrics_three_returns(self):
        metrics = ["Classification", "Proba_0", "Proba_1"]
        returns = ["", int, int]
        self.sc.score_code += (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.sc._binary_target(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [["A", 0.9, 0.1]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("A", 0.9, 0.1))
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [["A", 0.9, 0.1], ["B", 0.2, 0.8]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {
                    "Classification": ["A", "B"],
                    "Proba_0": [0.9, 0.2],
                    "Proba_1": [0.1, 0.8],
                }
            ),
        )


class TestNonbinaryTargets(unittest.TestCase):
    def setUp(self):
        self.sc = ScoreCode
        self.sc.score_code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def test_snippet(input_array, prediction):\n"
        )
        self.target_values = ["A", "B", "C"]

    def tearDown(self):
        self.sc.score_code = ""

    def execute_snippet(self, *args):
        scope = {}
        exec(self.sc.score_code, scope)
        test_snippet = scope["test_snippet"]
        return test_snippet(*args)

    def test_one_metric_h2o(self):
        metrics = "Classification"
        returns = ["", int, int]
        self.sc._nonbinary_targets(metrics, self.target_values, returns, h2o_model=True)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[], ["", 0.1, 0.2, 0.7]]
        self.assertEqual(self.execute_snippet(input_array, prediction), "C")
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = pd.DataFrame(
            {"predict": [0, 1], "p0": [0.8, 0.2], "p1": [0.1, 0.6], "p2": [0.1, 0.2]}
        )
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "B"]}),
        )

    def test_one_metric_one_return(self):
        metrics = "Classification"
        returns = [""]
        self.sc._nonbinary_targets(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = ["C"]
        self.assertEqual(self.execute_snippet(input_array, prediction), "C")
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = ["D", "J"]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["D", "J"]}),
        )

    def test_one_metric_probability_returns(self):
        metrics = "Classification"
        returns = [int, int, int]
        self.sc._nonbinary_targets(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0.1, 0.2, 0.3]]
        self.assertEqual(self.execute_snippet(input_array, prediction), "C")
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0.9, 0.8, 0.7], [0.4, 0.5, 0.1]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "B"]}),
        )

    def test_one_metric_classification_and_probability_returns(self):
        metrics = "Classification"
        returns = ["", int, int, int]
        self.sc._nonbinary_targets(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [["C", 0.1, 0.2, 0.3]]
        self.assertEqual(self.execute_snippet(input_array, prediction), "C")
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [["A", 0.9, 0.8, 0.7], ["C", 0.4, 0.5, 0.1]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "C"]}),
        )

    def test_two_metrics_h2o(self):
        metrics = ["Classification", "Probability"]
        returns = ["", int, int]
        self.sc._nonbinary_targets(metrics, self.target_values, returns, h2o_model=True)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[], ["", 0.1, 0.2, 0.7]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("C", 0.7))
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = pd.DataFrame(
            {"predict": [0, 1], "p0": [0.8, 0.2], "p1": [0.1, 0.6], "p2": [0.1, 0.2]}
        )
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "B"], "Probability": [0.8, 0.6]}),
        )

    def test_two_metrics_return_probabilities(self):
        metrics = ["Classification", "Probability"]
        returns = [int, int, int]
        self.sc._nonbinary_targets(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0.1, 0.2, 0.3]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("C", 0.3))
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0.9, 0.8, 0.7], [0.4, 0.5, 0.1]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "B"], "Probability": [0.9, 0.5]}),
        )

    def test_two_metrics_return_classification_and_probability(self):
        metrics = ["Classification", "Probability"]
        returns = ["", int, int, int]
        self.sc._nonbinary_targets(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [["C", 0.1, 0.2, 0.3]]
        self.assertEqual(self.execute_snippet(input_array, prediction), ("C", 0.3))
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [["A", 0.9, 0.8, 0.7], ["C", 0.4, 0.5, 0.1]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame({"Classification": ["A", "C"], "Probability": [0.9, 0.5]}),
        )

    def test_probability_metrics_h2o(self):
        metrics = ["Proba_1", "Proba_2", "Proba_3"]
        returns = []
        self.sc._nonbinary_targets(metrics, self.target_values, returns, h2o_model=True)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[], ["", 0.1, 0.2, 0.7]]
        self.assertEqual(self.execute_snippet(input_array, prediction), (0.1, 0.2, 0.7))
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = pd.DataFrame(
            {"predict": [0, 1], "p0": [0.8, 0.2], "p1": [0.1, 0.6], "p2": [0.1, 0.2]}
        )
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {"Proba_1": [0.8, 0.2], "Proba_2": [0.1, 0.6], "Proba_3": [0.1, 0.2]}
            ),
        )

    def test_classification_probability_metrics_h2o(self):
        metrics = ["Classification", "Prob1", "Prob2", "Prob3"]
        returns = []
        self.sc._nonbinary_targets(metrics, self.target_values, returns, h2o_model=True)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[], ["L", 0.1, 0.2, 0.7]]
        self.assertEqual(
            self.execute_snippet(input_array, prediction), ("L", 0.1, 0.2, 0.7)
        )
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = pd.DataFrame(
            {"predict": [0, 1], "p0": [0.8, 0.2], "p1": [0.1, 0.6], "p2": [0.1, 0.2]}
        )
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {
                    "Classification": [0, 1],
                    "Prob1": [0.8, 0.2],
                    "Prob2": [0.1, 0.6],
                    "Prob3": [0.1, 0.2],
                }
            ),
        )

    def test_return_all_probabilities(self):
        metrics = ["Proba_0", "Proba_1", "Proba_2"]
        returns = [int, int, int]
        self.sc._nonbinary_targets(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0.1, 0.2, 0.3]]
        self.assertEqual(self.execute_snippet(input_array, prediction), (0.1, 0.2, 0.3))
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0.9, 0.8, 0.7], [0.4, 0.5, 0.1]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {"Proba_0": [0.9, 0.4], "Proba_1": [0.8, 0.5], "Proba_2": [0.7, 0.1]}
            ),
        )

    def test_return_all_probabilities_and_classification(self):
        metrics = ["Classification", "Proba_0", "Proba_1", "Proba_2"]
        returns = ["", int, int, int]
        self.sc._nonbinary_targets(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [["P", 0.1, 0.2, 0.3]]
        self.assertEqual(
            self.execute_snippet(input_array, prediction), ("P", 0.1, 0.2, 0.3)
        )
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [["D", 0.9, 0.8, 0.7], ["C", 0.4, 0.5, 0.1]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {
                    "Classification": ["D", "C"],
                    "Proba_0": [0.9, 0.4],
                    "Proba_1": [0.8, 0.5],
                    "Proba_2": [0.7, 0.1],
                }
            ),
        )

    def test_return_all_probabilities_generate_classification(self):
        metrics = ["Classification", "Proba_0", "Proba_1", "Proba_2"]
        returns = [int, int, int]
        self.sc._nonbinary_targets(metrics, self.target_values, returns)
        # Single row
        input_array = pd.DataFrame([[1]], columns=["A"], index=[0])
        prediction = [[0.1, 0.2, 0.3]]
        self.assertEqual(
            self.execute_snippet(input_array, prediction), ("C", 0.1, 0.2, 0.3)
        )
        # Multiple rows
        input_array = pd.DataFrame({"A": [1, 0, 1]})
        prediction = [[0.9, 0.8, 0.7], [0.4, 0.5, 0.1]]
        pd.testing.assert_frame_equal(
            self.execute_snippet(input_array, prediction),
            pd.DataFrame(
                {
                    "Classification": ["A", "B"],
                    "Proba_0": [0.9, 0.4],
                    "Proba_1": [0.8, 0.5],
                    "Proba_2": [0.7, 0.1],
                }
            ),
        )


def test_predictions_to_metrics():
    """
    Test Cases:
    - flattened list -> len(score_metrics) == 1
    - no target_values or no thresholds
    - len(target_values) > 1
        - warning if threshold provided
    - target_values == 1
    - raise error for binary model w/ nonbinary targets
    - raise error for no target_values, but thresholds provided
    """
    with patch("sasctl.pzmm.ScoreCode._no_targets_no_thresholds") as func:
        metrics = ["Classification"]
        returns = [1]
        sc._predictions_to_metrics(metrics, returns)
        func.assert_called_once_with("Classification", returns, False)

    with patch("sasctl.pzmm.ScoreCode._nonbinary_targets") as func:
        target_values = ["A", "B", 5]
        sc._predictions_to_metrics(metrics, returns, target_values)
        func.assert_called_once_with("Classification", target_values, returns, False)

    with patch("sasctl.pzmm.ScoreCode._binary_target") as func:
        metrics = ["Classification", "Probability"]
        target_values = ["1", "0"]
        sc._predictions_to_metrics(metrics, returns, target_values)
        func.assert_called_once_with(metrics, ["1", "0"], returns, None, 1, False)

    with pytest.raises(
        ValueError,
        match="A threshold was provided to interpret the prediction results, however "
        "a target value was not, therefore, a valid output cannot be generated.",
    ):
        sc._predictions_to_metrics(metrics, returns, predict_threshold=0.7)


def test_input_var_lists():
    """
    Test Cases:
    - var list & dtype list
        - default
        - mlflow
    """
    data = pd.DataFrame(data=[[1, "A"], [5, "B"]], columns=["First", "Second"])
    var_list, dtypes_list = sc._input_var_lists(data)
    assert var_list == ["First", "Second"]
    assert dtypes_list == ["int64", "object"]

    data = [{"name": "First", "type": "int"}, {"name": "Second", "type": "string"}]
    var_list, dtypes_list = sc._input_var_lists(data)
    assert var_list == ["First", "Second"]
    assert dtypes_list == ["int", "string"]


@patch("sasctl.pzmm.ScoreCode._get_model_id")
@patch("sasctl.core.Session.version_info")
def test_check_viya_version(mock_version, mock_get_model):
    """
    Test Cases:
    - check Viya version
        - Viya 3.5
        - Viya 4
        - No connection
    """
    current_session(None)
    mock_version.return_value = None
    model = {"name": "Test", "id": "abc123"}
    with pytest.warns():
        assert sc._check_viya_version(model) is None

    with patch("sasctl.core.Session._get_authorization_token"):
        current_session("example.com", "user", "password")

    mock_version.return_value = VersionInfo(3, 5)
    with pytest.raises(SystemError):
        sc._check_viya_version(None)

    mock_get_model.return_value = model["id"]
    assert sc._check_viya_version(model) == "abc123"

    mock_version.return_value = VersionInfo(4)
    assert sc._check_viya_version(None) is None
    assert sc._check_viya_version(model) is None


def test_check_valid_model_prefix():
    """
    Test Cases:
    - check model_prefix validity
        - raise warning and replace if invalid
    """
    assert sc._check_valid_model_prefix("TestPrefix") == "TestPrefix"
    assert sc._check_valid_model_prefix("Test Prefix") == "Test_Prefix"


def test_write_score_code(score_code_mocks):
    """
    Test Cases:
    - set model_file_name
        - both; raise error
        - neither; raise error
        - model_file_name
        - binary_string
    - model_load cases
        - binary_string
        - Viya 3.5
        - Viya 4
    - create score file if score_code_path provided
        - return dict if not
    """
    score_code_mocks["_check_viya_version"].return_value = None
    score_code_mocks["_input_var_lists"].return_value = (["A", "B"], ["str", "int"])
    score_code_mocks["_viya35_model_load"].return_value = "3.5"
    score_code_mocks["_viya4_model_load"].return_value = "4"
    score_code_mocks["_viya35_score_code_import"].return_value = ("MAS", "CAS")
    score_code_mocks["_check_valid_model_prefix"].return_value = "TestModel"

    # No binary string or model file provided
    with pytest.raises(ValueError):
        sc.write_score_code(
            "TestModel",
            pd.DataFrame(data=[["A", 1], ["B", 2]], columns=["First", "Second"]),
            [predict_proba, []],
        )

    # Binary string and model file provided
    with pytest.raises(ValueError):
        sc.write_score_code(
            "TestModel",
            pd.DataFrame(data=[["A", 1], ["B", 2]], columns=["First", "Second"]),
            [predict_proba, []],
            model_file_name="model.pickle",
            binary_string=b"Binary model string.",
        )

    sc.write_score_code(
        "TestModel",
        pd.DataFrame(data=[["A", 1], ["B", 2]], columns=["First", "Second"]),
        [predict_proba, []],
        model_file_name="model.pickle",
    )
    score_code_mocks["_viya4_model_load"].assert_called_once()

    score_code_mocks["_check_viya_version"].return_value = "abc123"
    sc.write_score_code(
        "TestModel",
        pd.DataFrame(data=[["A", 1], ["B", 2]], columns=["First", "Second"]),
        [predict_proba, []],
        model_file_name="model.pickle",
    )
    score_code_mocks["_viya35_model_load"].assert_called_once()

    output_dict = sc.write_score_code(
        "TestModel",
        pd.DataFrame(data=[["A", 1], ["B", 2]], columns=["First", "Second"]),
        [predict_proba, []],
        binary_string=b"Binary model string.",
    )
    assert "score_TestModel.py" in output_dict
    assert "dmcas_packagescorecode.sas" in output_dict
    assert "dmcas_epscorecode.sas" in output_dict

    tmp_dir = tempfile.TemporaryDirectory()
    sc.write_score_code(
        "TestModel",
        pd.DataFrame(data=[["A", 1], ["B", 2]], columns=["First", "Second"]),
        [predict_proba, []],
        score_code_path=Path(tmp_dir.name),
        binary_string=b"Binary model string.",
    )
    assert (Path(tmp_dir.name) / "dmcas_packagescorecode.sas").exists()
    assert (Path(tmp_dir.name) / "dmcas_epscorecode.sas").exists()
    assert (Path(tmp_dir.name) / "score_TestModel.py").exists()
