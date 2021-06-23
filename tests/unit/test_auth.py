#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from six.moves import mock

from sasctl import Session
from sasctl.exceptions import AuthenticationError


ACCESS_TOKEN = 'abc123'
REFRESH_TOKEN = 'xyz'


def test_username_password_success():
    """Successful authentication with username & password."""

    with mock.patch('sasctl.core.requests.Session.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'access_token': ACCESS_TOKEN, 'refresh_token': REFRESH_TOKEN}

        s = Session('hostname', 'username', 'password')

        # POST data
        data = mock_post.call_args[1]['data']
        assert data['grant_type'] == 'password'
        assert data['username'] == 'username'
        assert data['password'] == 'password'

        assert s.auth.access_token == ACCESS_TOKEN
        assert s.auth.refresh_token == REFRESH_TOKEN


def test_username_password_failure():
    """Authentication failure with username & password should raise an exception."""

    with mock.patch('sasctl.core.requests.Session.post') as mock_post:
        mock_post.return_value.status_code = 401
        mock_post.return_value.json.return_value = {'access_token': ACCESS_TOKEN, 'refresh_token': REFRESH_TOKEN}

        with pytest.raises(AuthenticationError):
            s = Session('hostname', 'username', 'password')


def test_existing_token():
    """If an explicit token is provided it should be passed along."""

    s = Session('hostname', token=ACCESS_TOKEN)
    assert s.auth.access_token == ACCESS_TOKEN
    assert s.auth.refresh_token is None


def test_auth_code():
    """No username & password and no kerberos should fall back to auth code."""

    AUTH_CODE = 'supersecretauthcode'

    # Kerberos auth has to fail before auth code will be attempted
    with mock.patch('sasctl.core.Session._get_token_with_kerberos', side_effect=ValueError):

        with mock.patch('sasctl.core.input', return_value=AUTH_CODE):

            with mock.patch('sasctl.core.requests.Session.post') as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {'access_token': ACCESS_TOKEN,
                                                            'refresh_token': REFRESH_TOKEN}

                s = Session('hostname')

                # POST data
                data = mock_post.call_args[1]['data']
                assert data['grant_type'] == 'authorization_code'
                assert data['code'] == AUTH_CODE
                assert s.auth.access_token == ACCESS_TOKEN
                assert s.auth.refresh_token == REFRESH_TOKEN
