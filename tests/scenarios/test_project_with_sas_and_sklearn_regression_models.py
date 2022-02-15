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

sklearn = pytest.importorskip('sklearn')


# Every test function in the module will automatically receive the session fixture
pytestmark = pytest.mark.usefixtures('session')

SAS_MODEL_NAME = 'sasctl_test_SAS_Boston_regression'
SCIKIT_MODEL_NAME = 'sasctl_test_scikit_Boston_regression'
PROJECT_NAME = 'sasctl_test_Boston_Housing'


@pytest.fixture(autouse=True)
def run_around_tests(session):
    # Run setup/teardown code
    def clean():
        mr.delete_project(PROJECT_NAME)

    clean()
    yield
    clean()


def test(cas_session, boston_dataset):
    cas_session.loadactionset('regression')

    tbl = cas_session.upload(boston_dataset).casTable
    features = tbl.columns[tbl.columns != 'Price']

    # Fit a linear regression model in CAS and output an ASTORE
    tbl.glm(target='Price', inputs=list(features), savestate='model_table')
    astore = cas_session.CASTable('model_table')

    from sklearn.linear_model import LinearRegression
    X = boston_dataset.drop('Price', axis=1)
    y = boston_dataset['Price']
    sk_model = LinearRegression()
    sk_model.fit(X, y)

    sas_model = register_model(astore, SAS_MODEL_NAME, PROJECT_NAME, force=True)
    sk_model = register_model(sk_model, SCIKIT_MODEL_NAME, PROJECT_NAME, input=X)

    # Test overwriting model content
    mr.add_model_content(sk_model, 'Your mother was a hamster!', 'insult.txt')
    mr.add_model_content(sk_model, 'And your father smelt of elderberries!', 'insult.txt')

    # Publish to MAS
    sas_module = publish_model(sas_model, 'maslocal', replace=True)
    sk_module = publish_model(sk_model, 'maslocal', replace=True)

    # Pass a row of data to MAS and receive the predicted result.
    first_row = tbl.head(1)
    result = sas_module.score(first_row)
    assert isinstance(result, float)

    result2 = sk_module.predict(first_row)
    assert isinstance(result2, float)

    assert round(result, 5) == round(result2, 5)
