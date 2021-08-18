#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


@pytest.fixture
def change_dir():
    """Change working directory for the duration of the test."""
    old_dir = os.path.abspath(os.curdir)

    def change(new_dir):
        os.chdir(new_dir)

    yield change
    os.chdir(old_dir)


def test_astore_model(session, cas_session, change_dir):
    """Ensure the register_sas_classification_model.py example executes successfully."""

    # Mock up Session() to return the Betamax-recorded session
    def Session(*args, **kwargs):
        return session

    change_dir('examples')
    with open('register_sas_classification_model.py') as f:
        # Remove Session import and set CAS session to Betamax-recorded CAS
        # session
        code = f.read().replace('from sasctl import Session', '')
        code = code.replace(
            "s = swat.CAS('hostname', 5570, 'username', 'password')", 's = cas_session'
        )
        # Exec the script.
        exec(code)


def test_register_sas_regression_model(session, cas_session, change_dir):
    """Ensure the register_sas_regression_model.py example executes successfully."""

    # Mock up Session() to return the Betamax-recorded session
    def Session(*args, **kwargs):
        return session

    change_dir('examples')
    with open('register_sas_regression_model.py') as f:
        # Remove Session import and set CAS session to Betamax-recorded CAS
        # session
        code = f.read().replace('from sasctl import Session', '')
        code = code.replace(
            "with swat.CAS('hostname', 5570, 'username', 'password') as cas:",
            "with cas_session as cas:",
        )
        # Exec the script.
        exec(code)


def test_sklearn_model(session, change_dir):
    """Ensure the register_scikit_classification_model.py example executes successfully."""

    # Mock up Session() to return the Betamax-recorded session
    def Session(*args, **kwargs):
        return session

    change_dir('examples')
    with open('register_scikit_classification_model.py') as f:
        # Remove import of Session to ensure mock function will be used
        # instead.
        code = f.read().replace(
            'from sasctl import Session, register_model',
            'from sasctl import register_model',
        )
        exec(code)


def test_scikit_regression_model(session, change_dir):
    """Ensure the register_scikit_regression_model.py example executes successfully."""

    pytest.skip('Re-enable once MAS publish no longer hangs.')
    # Mock up Session() to return the Betamax-recorded session
    def Session(*args, **kwargs):
        return session

    change_dir('examples')
    with open('register_scikit_regression_model.py') as f:
        # Remove import of Session to ensure mock function will be used
        # instead.
        code = f.read().replace(
            'from sasctl import Session, register_model, publish_model',
            'from sasctl import register_model, publish_model',
        )
        exec(code)


def test_full_lifecycle(session, change_dir):
    """Ensure the register_scikit_classification_model.py example executes successfully."""

    pytest.skip(
        "Fix/re-implement.  Performance upload creates unrecorded CAS "
        "session that can't be replayed."
    )

    # Mock up Session() to return the Betamax-recorded session
    def Session(*args, **kwargs):
        return session

    change_dir('examples')
    with open('full_lifecycle.py') as f:
        # Remove import of Session to ensure mock function will be used
        # instead.
        code = f.read().replace('from sasctl import Session', '')

        exec(code)


def test_direct_rest_calls(session, change_dir):
    """Ensure the direct_REST_calls.py example executes successfully."""
    from pickle import UnpicklingError

    # Mock up Session() to return the Betamax-recorded session
    def Session(*args, **kwargs):
        return session

    change_dir('examples')
    with open('direct_REST_calls.py') as f:
        # Remove import of Session to ensure mock function will be used
        # instead.
        code = f.read().replace(
            'from sasctl import get, get_link, request_link, Session',
            'from sasctl import get, get_link, request_link',
        )
        try:
            exec(code)
        except (UnpicklingError, KeyError) as e:
            if "'\xef'" in str(e):
                # Betamax recording adds additional bytes to the content which
                # breaks unpickling.  Ignore when this happens as correct
                # handling of binary contents should be validated in integration
                # tests
                pass


def test_register_custom_model(session, change_dir):
    """Ensure the register_custom_model.py example executes successfully."""
    from pickle import UnpicklingError

    # Mock up Session() to return the Betamax-recorded session
    def Session(*args, **kwargs):
        return session

    change_dir('examples')
    with open('register_custom_model.py') as f:
        # Remove import of Session to ensure mock function will be used
        # instead.
        code = f.read().replace(
            'from sasctl import register_model, Session',
            'from sasctl import register_model',
        )
        try:
            exec(code)
        except (UnpicklingError, KeyError) as e:
            if "'\xef'" in str(e):
                # Betamax recording adds additional bytes to the content which
                # breaks unpickling.  Ignore when this happens as correct
                # handling of binary contents should be validated in integration
                # tests
                pass
