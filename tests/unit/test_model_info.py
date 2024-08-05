#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sasctl.utils.model_info import get_model_info


@pytest.mark.parametrize(
    "model, algorithm",
    [
        (LinearRegression(), "Linear regression"),
        (DecisionTreeRegressor(), "Decision tree"),
        (RandomForestRegressor(), "Forest"),
        (GradientBoostingRegressor(), "Gradient boosting"),
        (SVR(), "Support vector machine"),
    ],
)
def test_sklearn_regression(boston_dataset, model, algorithm):
    """Verify correct ModelInfo properties for scikit-learn regression algorithms."""
    target = "Price"
    X = boston_dataset.drop(columns=target)
    y = boston_dataset[target]
    model.fit(X, y)
    info = get_model_info(model, X, y)

    assert info.is_regressor
    assert not info.is_classifier
    assert not info.is_binary_classifier
    assert not info.is_clusterer
    assert info.analytic_function == "prediction"
    assert info.algorithm == algorithm
    assert info.output_column_names == [target]
    assert info.target_values is None
    assert info.predict_function == model.predict


def test_sklearn_regression_target_column_name(boston_dataset):
    """Verify output column name set correctly when target frame has no column name."""
    target = "Price"
    X = boston_dataset.drop(columns=target)

    # Use numpy array with no column names
    y = boston_dataset[target].values

    model = LinearRegression().fit(X, y)
    info = get_model_info(model, X, y)

    assert info.output_column_names == ["I_Target"]


@pytest.mark.parametrize(
    "model, algorithm",
    [
        (LogisticRegression(), "Logistic regression"),
        (DecisionTreeClassifier(), "Decision tree"),
        (RandomForestClassifier(), "Forest"),
        (GradientBoostingClassifier(), "Gradient boosting"),
        (SVC(probability=True), "Support vector machine"),
    ],
)
def test_sklearn_binary_classifier(cancer_dataset, model, algorithm):
    target = "Type"
    X = cancer_dataset.drop(columns=target)
    y = cancer_dataset[target]
    model.fit(X, y)
    info = get_model_info(model, X, y)

    assert info.is_classifier
    assert info.is_binary_classifier
    assert not info.is_regressor
    assert not info.is_clusterer
    assert info.analytic_function == "classification"
    assert info.algorithm == algorithm
    assert info.output_column_names == [target]  # target_variable
    assert info.target_values == ["malignant"]  # target_event
    assert info.predict_function == model.predict

    # If output frame contains a column name then it should be retained
    y = pd.DataFrame(y, columns=[target])
    info = get_model_info(model, X, y)
    assert info.output_column_names == [target]

    # If output frame contains class probabilities then column names should reflect classes
    y_probs = model.predict_proba(X)
    info = get_model_info(model, X, y_probs)
    assert info.output_column_names == ["P_benign", "P_malignant"]
    assert info.is_binary_classifier
    assert info.threshold == 0.5

    # assume predict function -
    """
    if no target value & no predict threshold & predict function is classification, assume model response is just the label

    if output_column_names = [I_, P_, _P] but nothing says what columns mean, assume .predict(data) returns [I_, P_, P_]

    if only 1 target value then assumed to be binary
    if assumed to be binary then 

    if output is single column
    target_values is class labels for corresponding probabilities (assuming output_columns is P_, P_, P_)



    """


def test_sklearn_binary_classifier_target_column_name(cancer_dataset):
    """Verify output column name set correctly when target frame has no column name."""
    target = "Type"
    X = cancer_dataset.drop(columns=target)
    y = cancer_dataset[target].values

    model = LogisticRegression().fit(X, y)
    info = get_model_info(model, X, y)

    assert info.output_column_names == ["I_Target"]


@pytest.mark.parametrize(
    "model, algorithm",
    [
        (LogisticRegression(), "Logistic regression"),
        (DecisionTreeClassifier(), "Decision tree"),
        (RandomForestClassifier(), "Forest"),
        (GradientBoostingClassifier(), "Gradient boosting"),
        (SVC(probability=True), "Support vector machine"),
    ],
)
def test_sklearn_multiclass_classifier(iris_dataset, model, algorithm):
    target = "Species"
    X = iris_dataset.drop(columns=target)
    y = iris_dataset[target]
    model.fit(X, y)
    info = get_model_info(model, X, y)

    assert info.is_classifier
    assert not info.is_binary_classifier
    assert not info.is_regressor
    assert not info.is_clusterer
    assert info.analytic_function == "classification"
    assert info.algorithm == algorithm
    assert info.output_column_names == [target]
    assert info.target_values == ["setosa", "versicolor", "virginica"]
    assert info.predict_function == model.predict

    # If output frame contains a column name then it should be retained
    y = pd.DataFrame(y, columns=[target])
    info = get_model_info(model, X, y)
    assert info.output_column_names == [target]

    # If output frame contains class probabilities then column names should reflect classes
    y_probs = model.predict_proba(X)
    info = get_model_info(model, X, y_probs)
    assert info.output_column_names == ["P_setosa", "P_versicolor", "P_virginica"]
    assert info.threshold is None

    # If output frame of probabilities somehow has column names, those should be retained
    y_probs = pd.DataFrame(y_probs, columns=["P1", "P2", "P3"])
    info = get_model_info(model, X, y_probs)
    assert info.output_column_names == list(y_probs.columns)
