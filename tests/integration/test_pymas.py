#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import warnings

import pytest


def dummy_function(x1, x2):
    # type: (float, float) -> (float, float)
    return x1+x2, x1-x2

@pytest.fixture
def train_data():
    """Returns the Iris data set as (X, y) """

    try:
        import pandas as pd
    except ImportError:
        pytest.skip('Package `pandas` not found.')

    try:
        from sklearn import datasets
    except ImportError:
        pytest.skip('Package `sklearn` not found.')

    raw = datasets.load_iris()
    iris = pd.DataFrame(raw.data, columns=raw.feature_names)
    iris = iris.join(pd.DataFrame(raw.target))
    iris.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    iris['Species'] = iris['Species'].astype('category')
    iris.Species.cat.categories = raw.target_names
    return iris.iloc[:, 0:4], iris['Species']


@pytest.fixture
def sklearn_model(train_data):
    """Returns a simple Scikit-Learn model """

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip('Package `sklearn` not found.')

    X, y = train_data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(X, y)
    return model


@pytest.fixture
def pickle_file(tmpdir_factory, sklearn_model):
    """Returns the path to a file containing a pickled Scikit-Learn model """
    import os

    sklearn_model
    filename = str(tmpdir_factory.mktemp('models').join('model.pkl'))
    with open(filename, 'wb') as f:
        pickle.dump(sklearn_model, f)
    yield str(filename)
    os.remove(filename)


@pytest.fixture
def python_file(tmpdir_factory):
    """Returns the path to a file containing source code for a Python function."""
    import os

    filename = str(tmpdir_factory.mktemp('models').join('model.py'))
    with open(filename, 'w') as f:
        f.writelines(['def predict(a, b, c, d):\n',
                      '    # types: (int, int, int, int) -> double\n'
                      '    pass\n'])

    yield str(filename)
    os.remove(filename)


@pytest.fixture
def pickle_stream(sklearn_model):
    """Returns a byte stream containing a pickled Scikit-Learn model """

    return pickle.dumps(sklearn_model)


def test_from_inline():
    from sasctl.utils.pymas import from_inline, PyMAS

    p = from_inline(dummy_function)
    assert isinstance(p, PyMAS)


def test_from_pickle(train_data, pickle_file):
    from sasctl.utils.pymas import PyMAS, from_pickle

    X, y = train_data
    p = from_pickle(pickle_file, func_name='predict', input_types=X)

    assert isinstance(p, PyMAS)


def test_from_pickle_without_dill(train_data, pickle_file, monkeypatch):
    import sasctl
    from sasctl.utils.pymas import PyMAS, from_pickle

    X, y = train_data

    # Ensure dill module is unavailable
    monkeypatch.setattr(sasctl.utils.pymas.core, 'dill', None)
    p = from_pickle(pickle_file, func_name='predict', input_types=X)

    assert isinstance(p, PyMAS)
    assert ' = pickle.loads(' in p.score_code()
    assert ' = dill.loads(' not in p.score_code()


def test_from_pickle_stream(train_data, pickle_stream):
    from sasctl.utils.pymas import PyMAS, from_pickle

    X, y = train_data
    p = from_pickle(pickle_stream, func_name='predict', input_types=X)
    assert isinstance(p, PyMAS)


def test_from_python_file(python_file):
    from sasctl.utils.pymas import PyMAS, from_python_file

    p = from_python_file(python_file, func_name='predict')
    assert isinstance(p, PyMAS)

