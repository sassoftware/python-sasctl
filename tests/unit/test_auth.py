#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from sasctl import Session
from sasctl.exceptions import AuthenticationError


ACCESS_TOKEN = 'abc123'
REFRESH_TOKEN = 'xyz'


def test_username_password_success():
    """Successful authentication with username & password."""

    with mock.patch('sasctl.core.requests.Session.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'access_token': ACCESS_TOKEN,
            'refresh_token': REFRESH_TOKEN,
        }

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
        mock_post.return_value.json.return_value = {
            'access_token': ACCESS_TOKEN,
            'refresh_token': REFRESH_TOKEN,
        }

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
    with mock.patch(
        'sasctl.core.Session._get_token_with_kerberos', side_effect=ValueError
    ):

        # Dont read anything from disk
        with mock.patch('sasctl.core.Session.read_cached_token', return_value=None):

            # Don't actually prompt user to input auth code
            with mock.patch('sasctl.core.input', return_value=AUTH_CODE):

                # Don't write the fake token to disk
                with mock.patch('sasctl.core.Session.cache_token'):

                    with mock.patch('sasctl.core.requests.Session.post') as mock_post:
                        mock_post.return_value.status_code = 200
                        mock_post.return_value.json.return_value = {
                            'access_token': ACCESS_TOKEN,
                            'refresh_token': REFRESH_TOKEN,
                        }

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
    target = {
        'profiles': [
            {
                'baseurl': 'https://example.sas.com',
                'name': 'Example',
                'oauthtoken': {
                    'accesstoken': 'abc123',
                    'expiry': None,
                    'refreshtoken': 'xyz',
                    'tokentype': 'bearer',
                },
            }
        ]
    }

    # Fake file exists
    with mock.patch('os.path.exists', return_value=True):

        # Fake permissions on a (fake) file
        with mock.patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = 0o600

            # Open & read fake file
            with mock.patch('builtins.open', mock.mock_open(read_data=fake_yaml)):
                tokens = Session._read_token_cache(Session.PROFILE_PATH)

    assert tokens == target


def test_write_token_cache():
    """Test writing tokens in YAML format to disk."""
    profiles = {
        'profiles': [
            {
                'baseurl': 'https://example.sas.com',
                'name': 'Example',
                'oauthtoken': {
                    'accesstoken': 'abc123',
                    'expiry': None,
                    'refreshtoken': 'xyz',
                    'tokentype': 'bearer',
                },
            }
        ]
    }

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
    assert handle.write.call_count > 0  # called for each line of yaml written


def test_automatic_token_refresh():
    """Access token should automatically be refreshed on an HTTP 401 indicating expired token."""
    from sasctl.core import OAuth2Token

    with mock.patch('sasctl.core.requests.Session.request') as mock_request:
        mock_request.return_value.status_code = 401
        mock_request.return_value.headers = {
            'WWW-Authenticate': 'Bearer realm="oauth", error="invalid_token", error_description="Access token expired: abc"'
        }

        with mock.patch(
            'sasctl.core.Session.get_auth',
            return_value=OAuth2Token(access_token='abc', refresh_token='def'),
        ):
            s = Session('example.com')

        assert s.auth.access_token == 'abc'
        assert s.auth.refresh_token == 'def'

        with mock.patch(
            'sasctl.core.Session.get_oauth_token',
            return_value=OAuth2Token(access_token='uvw', refresh_token='xyz'),
        ) as mock_oauth:
            s.get('/fakeurl')

        assert s.auth.access_token == 'uvw'
        assert s.auth.refresh_token == 'xyz'


def test_load_expired_token_with_refresh():
    """If a cached token is loaded but it's expired it should be refreshed.

    1) Session is created with no credentials passed
    2) Token cache is checked for existing access token
    3) Cached token is found, but is expired
    4) Refresh token is used to acquire new access token
    5) New access token is returned and used by Session

    """
    from datetime import datetime, timedelta
    from sasctl.core import OAuth2Token

    # Cached profile with an expired access token
    PROFILES = {
        'profiles': [
            {
                'baseurl': 'https://example.com',
                'oauthtoken': {
                    'accesstoken': 'abc',
                    'refreshtoken': 'def',
                    'expiry': datetime.now() - timedelta(seconds=1),
                },
            }
        ]
    }

    # Return fake profiles instead of reading from disk
    with mock.patch('sasctl.core.Session._read_token_cache', return_value=PROFILES):

        # Fake response for refresh token request.
        with mock.patch(
            'sasctl.core.Session.get_oauth_token',
            return_value=OAuth2Token(access_token='xyz'),
        ):
            s = Session('example.com')

        # Cached token is expired, should have refreshed and gotten new token
        assert s.auth.access_token == 'xyz'


def test_load_expired_token_no_refresh():
    """If a cached token is loaded but it's expired and can't be refreshed, auth code prompt should be shown.

    1) Session is created with no credentials passed
    2) Token cache is checked for existing access token
    3) Cached token is found, but is expired
    4) Refresh token is used to acquire new access token
    5) Refresh token is found to be expired
    6) Cached tokens are ignored
    7) User is prompted for authorization code

    """
    from datetime import datetime, timedelta
    from sasctl.core import OAuth2Token
    from sasctl.exceptions import AuthorizationError

    # Cached profile with an expired access token
    PROFILES = {
        'profiles': [
            {
                'baseurl': 'https://example.com',
                'oauthtoken': {
                    'accesstoken': 'abc',
                    'refreshtoken': 'def',
                    'expiry': datetime.now() - timedelta(seconds=1),
                },
            }
        ]
    }

    # Return fake profiles instead of reading from disk
    with mock.patch('sasctl.core.Session._read_token_cache', return_value=PROFILES):

        # Fake an expired refresh token - AuthorizationError should be raised
        with mock.patch(
            'sasctl.core.Session.get_oauth_token', side_effect=AuthorizationError
        ):

            # Refresh of expired token failed, so user should be prompted for auth code
            with mock.patch('sasctl.core.Session.prompt_for_auth_code') as mock_prompt:

                with pytest.raises(AuthorizationError):
                    s = Session('example.com')

            assert mock_prompt.call_count == 1
