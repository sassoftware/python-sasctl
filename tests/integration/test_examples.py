#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import six

def test_astore_model(session, cas_session):
    """Ensure the astore_model.py example executes successfully."""

    pytest.xfail('Model publish fails.')

    # Mock up Session() to return the Betamax-recorded session
    def Session(*args, **kwargs):
        return session

    with open('examples/astore_model.py') as f:
        # Remove Session import and set CAS session to Betamax-recorded CAS
        # session
        code = f.read().replace('from sasctl import Session', '')
        code = code.replace("s = swat.CAS('hostname', 'username', "
                            "'password')",
                            's = cas_session')
        # Exec the script.
        six.exec_(code)


def test_sklearn_model(session):
    """Ensure the sklearn_model.py example executes successfully."""

    # Mock up Session() to return the Betamax-recorded session
    def Session(*args, **kwargs):
        return session

    with open('examples/sklearn_model.py') as f:
        # Remove import of Session to ensure mock function will be used
        # instead.
        code = f.read().replace('from sasctl import Session, register_model',
                                'from sasctl import register_model')
        six.exec_(code)