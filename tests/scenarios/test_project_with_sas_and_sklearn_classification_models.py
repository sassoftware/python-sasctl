#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register SAS and scikit regression models.

Performs the following steps:
 - Train a SAS GLM model on the Boston housing dataset
 - Train a scikit-learn LinearRegression model on the Boston housing dataset
 - Register the SAS model into a new project
 - Register the scikit-learn model into the same project
 - Publish the SAS model to MAS
 - Publish the scikit-learn model to MAS
 - Call both MAS modules with the same input row
 - Verify that results match

"""

import pytest

from sasctl import publish_model, register_model
from sasctl.services import model_repository as mr

sklearn = pytest.importorskip("sklearn")


# Every test function in the module will automatically receive the session fixture
pytestmark = pytest.mark.usefixtures("session")

SAS_MODEL_NAME = "sasctl_test_SAS_Iris_Gradboost"
SCIKIT_MODEL_NAME = "sasctl_test_scikit_Iris_Gradboost"
PROJECT_NAME = "sasctl_test_Iris_Species"
TARGET = "Species"


@pytest.fixture(autouse=True)
def run_around_tests(session):
    # Run setup/teardown code
    def clean():
        mr.delete_project(PROJECT_NAME)

    clean()
    yield
    clean()


def test(cas_session, iris_dataset):
    pytest.skip("Re-enable once MAS publish no longer hangs.")
    cas_session.loadactionset("decisiontree")

    tbl = cas_session.upload(iris_dataset).casTable
    features = list(tbl.columns[tbl.columns != TARGET])

    # Fit a linear regression model in CAS and output an ASTORE
    tbl.decisiontree.gbtreetrain(
        target=TARGET, inputs=features, savestate="model_table"
    )
    astore = cas_session.CASTable("model_table")

    from sklearn.ensemble import GradientBoostingClassifier

    X = iris_dataset.drop(TARGET, axis=1)
    y = iris_dataset[TARGET]
    sk_model = GradientBoostingClassifier()
    sk_model.fit(X, y)

    sas_model = register_model(astore, SAS_MODEL_NAME, PROJECT_NAME, force=True)
    sk_model = register_model(sk_model, SCIKIT_MODEL_NAME, PROJECT_NAME, input=X)

    # Publish to MAS
    sas_module = publish_model(sas_model, "maslocal", replace=True)
    sk_module = publish_model(sk_model, "maslocal", replace=True)

    # Pass a row of data to MAS and receive the predicted result.
    first_row = tbl.head(1)
    result = sas_module.score(first_row)
    p1, p1, p2, species, warning = result

    result2 = sk_module.predict(first_row)
    assert result2 in ("setosa", "virginica", "versicolor")

    # SAS model may have CHAR variable that's padded with spaces.
    assert species.strip() == result2

    result3 = sk_module.predict_proba(first_row)
    assert round(sum(result3), 5) == 1
