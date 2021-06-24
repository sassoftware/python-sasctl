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

        # Dont read anything from disk
        with mock.patch('sasctl.core.Session.read_cached_token', return_value=None):

            # Don't actually prompt user to input auth code
            with mock.patch('sasctl.core.input', return_value=AUTH_CODE):

                # Don't write the fake token to disk
                with mock.patch('sasctl.core.Session.cache_token'):

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


def test_read_cached_token():
    """Verify that cached tokens are read correctly"""

    # Example YAML file
    fake_yaml = """
profiles:
- baseurl: https://example.sas.com
  name: Example
  oauthtoken:
    accesstoken: abc123
    expiry: null
    refreshtoken: xyz
    tokentype: bearer
    """

    # Expected response
    target = {'profiles': [
        {'baseurl': 'https://example.sas.com',
         'name': 'Example',
         'oauthtoken': {
             'accesstoken': 'abc123',
             'expiry': None,
             'refreshtoken': 'xyz',
             'tokentype': 'bearer'
         }}
    ]}

    with mock.patch('sasctl.core.open', mock.mock_open(read_data=fake_yaml)):
        tokens = Session._read_token_cache(Session.PROFILE_PATH)

    assert tokens == target


def test_write_token_cache():
    """Test writing tokens in YAML format to disk."""
    profiles = {'profiles': [
        {'baseurl': 'https://example.sas.com',
         'name': 'Example',
         'oauthtoken': {
             'accesstoken': 'abc123',
             'expiry': None,
             'refreshtoken': 'xyz',
             'tokentype': 'bearer'
         }}
    ]}

    # Fake file object that will be written to
    mock_open = mock.mock_open()

    # Fake permissions on a (fake) file
    with mock.patch('os.stat') as mock_stat:
        mock_stat.return_value.st_mode = 0o600

        # Fake opening a file
        with mock.patch('sasctl.core.open', mock_open):
            Session._write_token_cache(profiles, Session.PROFILE_PATH)

    assert mock_open.call_count == 1
    handle = mock_open()
    handle.write.assert_called()  # called for each line of yaml written

