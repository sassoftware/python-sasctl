#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import pandas as pd

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sasctl.utils.model_info import get_model_info


@pytest.mark.parametrize("model, algorithm", [(LinearRegression(), "Linear regression"),
                                            (DecisionTreeRegressor(), "Decision tree"),
                                            (RandomForestRegressor(), "Forest"),
                                            (GradientBoostingRegressor(), "Gradient boosting"),
                                            (SVR(), "Support vector machine")])
def test_sklearn_regression(boston_dataset, model, algorithm):
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
    assert info.output_column_names == ["I_Target"]  # y is a series so has no column name
    assert info.target_values == None
    assert info.predict_function == model.predict



@pytest.mark.parametrize("model, algorithm", [(LogisticRegression(), "Logistic regression"),
                                            (DecisionTreeClassifier(), "Decision tree"),
                                            (RandomForestClassifier(), "Forest"),
                                            (GradientBoostingClassifier(), "Gradient boosting"),
                                            (SVC(probability=True), "Support vector machine")])
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
    assert info.output_column_names == ["I_Target"]  # y is a series so has no column name
    assert info.target_values == ["malignant"]
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


@pytest.mark.parametrize("model, algorithm", [(LogisticRegression(), "Logistic regression"),
                                            (DecisionTreeClassifier(), "Decision tree"),
                                            (RandomForestClassifier(), "Forest"),
                                            (GradientBoostingClassifier(), "Gradient boosting"),
                                            (SVC(probability=True), "Support vector machine")])
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
    assert info.output_column_names == ["I_Target"]  # y is a series so has no column name
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
