#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest import mock

import pytest
from sasctl import Session
from sasctl.core import OAuth2Token
from sasctl.exceptions import AuthenticationError, AuthorizationError


HOSTNAME = "example.sas.com"
USERNAME = "michael.palin"
PASSWORD = 'IAmALumberjack'
CLIENT_ID = "lumberjack"
CLIENT_SECRET = "ISleepAllNightAndWorkAllDay"
ACCESS_TOKEN = 'abc123'
REFRESH_TOKEN = 'xyz'


@mock.patch('sasctl.core.requests.Session.post')
def test_get_token_with_username_password_success(post):
    """Successful authentication with username & password."""
    post.return_value.status_code = 200
    post.return_value.json.return_value = {
        'access_token': ACCESS_TOKEN,
        'refresh_token': REFRESH_TOKEN,
    }

    s = Session('example.sas.com', USERNAME, PASSWORD)

    # Verify Session token is correctly set
    assert s.auth.access_token == ACCESS_TOKEN
    assert s.auth.refresh_token == REFRESH_TOKEN

    # Verify correct POST data was sent.
    assert post.call_count == 1
    url, args = post.call_args
    assert url[0] == 'https://example.sas.com/SASLogon/oauth/token#password'
    assert args['data']['grant_type'] == 'password'
    assert args['data']['username'] == USERNAME
    assert args['data']['password'] == PASSWORD
    assert args['auth'] == ('sas.ec', '')


@mock.patch('sasctl.core.requests.Session.post')
def test_get_token_with_username_password_failure(post):
    """Authentication failure with username & password should raise an exception."""
    post.return_value.status_code = 401
    post.return_value.json.return_value = {
        "error": "unauthorized",
        "error_description": "Bad credentials"
    }

    with pytest.raises(AuthenticationError):
        s = Session('hostname', 'username', 'password')


@mock.patch('sasctl.core.requests.Session.post')
def test_get_token_with_username_password_and_client(post):
    """Verify that custom client id/secret are used for password auth."""
    post.return_value.status_code = 401
    post.return_value.json.return_value = {
        "error": "unauthorized",
        "error_description": "Bad credentials"
    }

    with pytest.raises(AuthenticationError) as e:
        s = Session(HOSTNAME, USERNAME, PASSWORD, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    # Authentication error message should continue the username
    e.match(USERNAME)

    # Ensure we only made 1 call (for password authentication).  Should NOT have made a second attempt using
    # client credentials, even though they were provided.
    assert post.call_count == 1

    url, args = post.call_args
    assert url[0] == f'https://{HOSTNAME}/SASLogon/oauth/token#password'
    assert args['data']['grant_type'] == 'password'
    assert args['data']['username'] == USERNAME
    assert args['data']['password'] == PASSWORD
    assert args['auth'] == (CLIENT_ID, CLIENT_SECRET)


@mock.patch('sasctl.core.requests.Session.post')
def test_get_token_with_client_secret_success(post):
    """Verify correct REST call for authentication with a client secret argument."""
    post.return_value.status_code = 200

    # NOTE: refresh_token is not returned when using client credential authentication
    post.return_value.json.return_value = {
        'access_token': ACCESS_TOKEN,
        'token_type': 'bearer',
        'expires_in': 14399,
        'scope': 'uaa.none'
    }

    s = Session(HOSTNAME, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    # Verify Session token is correctly set
    assert s.auth.access_token == ACCESS_TOKEN
    assert s.auth.refresh_token is None

    # Verify correct POST data was sent.
    assert post.call_count == 1
    url, args = post.call_args
    assert url[0] == f'https://{HOSTNAME}/SASLogon/oauth/token#client_credentials'
    assert args['data']['grant_type'] == 'client_credentials'
    assert args['auth'] == (CLIENT_ID, CLIENT_SECRET)


@mock.patch('sasctl.core.requests.Session.post')
@mock.patch.dict(os.environ, {"SASCTL_CLIENT_ID": CLIENT_ID, "SASCTL_CLIENT_SECRET": CLIENT_SECRET})
def test_get_token_with_client_secret_env_var_success(post):
    """Verify correct REST call for authentication with a client secret from environment variables."""
    post.return_value.status_code = 200

    # NOTE: refresh_token is not returned when using client credential authentication
    post.return_value.json.return_value = {
        'access_token': ACCESS_TOKEN,
        'token_type': 'bearer',
        'expires_in': 14399,
        'scope': 'uaa.none'
    }

    s = Session(HOSTNAME)

    # Verify Session token is correctly set
    assert s.auth.access_token == ACCESS_TOKEN
    assert s.auth.refresh_token is None

    # Verify correct POST data was sent.
    assert post.call_count == 1
    url, args = post.call_args
    assert url[0] == f'https://{HOSTNAME}/SASLogon/oauth/token#client_credentials'
    assert args['data']['grant_type'] == 'client_credentials'
    assert args['auth'] == (CLIENT_ID, CLIENT_SECRET)


@mock.patch.dict(os.environ, {"SASCTL_CLIENT_ID": CLIENT_ID})
def test_get_token_with_client_secret_missing_secret():
    """An exception should be raised if client id provided but not client secret."""
    with pytest.raises(Exception) as e:
        s = Session(HOSTNAME)


@mock.patch('sasctl.core.requests.Session.post')
def test_get_token_with_client_secret_with_wrong_secret(post):
    """An exception should be raised if an invalid client id/secret is used."""
    post.return_value.status_code = 401
    post.return_value.json.return_value = {"error": "unauthorized", "error_description": "Bad credentials"}

    with pytest.raises(AuthenticationError) as e:
        s = Session(HOSTNAME, client_id=CLIENT_ID, client_secret='SomeInvalidSecret')

    e.match("Invalid client id or secret")


@mock.patch('sasctl.core.Session.read_cached_token')
@mock.patch('sasctl.core.Session.prompt_for_auth_code')
@mock.patch('sasctl.core.requests.Session.post')
def test_get_token_with_auth_code_success(post, prompt, token_cache):
    """If client credential authentication not allowed, auth code should be attempted."""

    AUTH_CODE = 'abcd'
    def fake_response(*args, **kwargs):
        # Generates a mock response to POST request
        m = mock.MagicMock()

        # If requesting client credential login, fake a failure
        if kwargs['data']['grant_type'] == 'client_credentials':
            m.status_code = 401
            m.json.return_value = {"error": "invalid_client",
                                   "error_description":"Unauthorized grant type: client_credentials"
                                   }

        # If requesting auth code login, accept
        elif kwargs['data']['grant_type'] == 'authorization_code':
            m.status_code = 200
            m.json.return_value = {
                'access_token': ACCESS_TOKEN,
                'refresh_token': REFRESH_TOKEN
            }

        # Shouldn't get a POST for anything but the initial client credentials and then the fallback request for
        # auth code login
        else:
            pytest.fail(f"Received unexpected POST arguments: {args}, {kwargs}.")

        return m

    # HTTP POST response needs to depend on whether it's client credentials or authorization code
    post.side_effect = fake_response

    # Need to ensure that no cached tokens can be returned
    token_cache.return_value = None

    # When prompted for an authorization code, respond.
    prompt.return_value = AUTH_CODE

    s = Session(HOSTNAME, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    # OAuth token should have been set
    assert s.auth.access_token == ACCESS_TOKEN
    assert s.auth.refresh_token == REFRESH_TOKEN

    # Verify that 2 calls were made, first for client credentials authentication and then again with an auth code
    assert post.call_count == 2
    assert '#client_credentials' in post.call_args_list[0][0][0]
    assert '#authorization_code' in post.call_args_list[1][0][0]

    # Verify that we tried to read a token from the cache before resorting to prompting the user.
    assert token_cache.called

    # Verify that user was prompted to enter the auth code & that correct client id was used to generate URL
    assert prompt.call_count == 1
    assert prompt.call_args[0][0] == CLIENT_ID


@mock.patch('sasctl.core.Session.read_cached_token')
@mock.patch('sasctl.core.Session.prompt_for_auth_code')
@mock.patch('sasctl.core.requests.Session.post')
def test_get_token_with_auth_code_failure(post, prompt, token_cache):
    """If client credential authentication not allowed, auth code should be attempted."""
    AUTH_CODE = 'abcd'
    def fake_response(*args, **kwargs):
        # Generates a mock response to POST request
        m = mock.MagicMock()

        # If requesting client credential login, fake a failure
        if kwargs['data']['grant_type'] == 'client_credentials':
            m.status_code = 401
            m.json.return_value = {"error": "invalid_client",
                                   "error_description":"Unauthorized grant type: client_credentials"
                                   }

        # If requesting auth code login, accept
        elif kwargs['data']['grant_type'] == 'authorization_code':
            m.status_code = 400
            m.json.return_value = {
                "error_description": "Invalid authorization code"
            }

        # Shouldn't get a POST for anything but the initial client credentials and then the fallback request for
        # auth code login
        else:
            pytest.fail(f"Received unexpected POST arguments: {args}, {kwargs}.")

        return m

    # HTTP POST response needs to depend on whether it's client credentials or authorization code
    post.side_effect = fake_response

    # Need to ensure that no cached tokens can be returned
    token_cache.return_value = None

    # When prompted for an authorization code, respond.
    prompt.return_value = AUTH_CODE

    with pytest.raises(AuthorizationError) as e:
        s = Session(HOSTNAME, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    # Verify that 2 calls were made, first for client credentials authentication and then again with an auth code
    assert post.call_count == 2
    assert '#client_credentials' in post.call_args_list[0][0][0]
    assert '#authorization_code' in post.call_args_list[1][0][0]

    # Verify that we tried to read a token from the cache before resorting to prompting the user.
    assert token_cache.called

    # Verify that user was prompted to enter the auth code & that correct client id was used to generate URL
    assert prompt.call_count == 1
    assert prompt.call_args[0][0] == CLIENT_ID


def test_get_token_with_consul():
    """Verify correct REST call for authentication with a Consul token."""
    pytest.fail('implement')


@mock.patch('sasctl.core.kerberos')
@mock.patch('sasctl.core.Session._request_token_with_kerberos')
@mock.patch('sasctl.core.requests.Session.post')
def test_auth_flow_username_only(post, get_token, kerberos):
    """If only a username is specified then just Kerberos authentication should be attempted."""

    # NOTE: don't need to do anything to kerberos package, just have to mock sasctl.core.kerberos so that variable
    #       is not None (code will assume package is imported)

    get_token.return_value = OAuth2Token(ACCESS_TOKEN)
    s = Session(HOSTNAME, USERNAME)

    # OAuth token should have been set
    assert s.auth.access_token == ACCESS_TOKEN

    # Make sure we made the function call to authenticate with kerberos
    # Call should have also included the username
    assert get_token.called
    assert get_token.call_args[0][0] == USERNAME  # TODO: Fix.  Bug causes username to be overwritten with None from .authinfo file

    # Ensure we did NOT try any other authentication flows.
    assert not post.called


# No username should call Kerberos (and update .username with user name pulled from Kerb)
# Username & no password should call kerberos
# username@realm should call kerberos...drop @REALM from username

def test_kerberos(kerberos):
    with mock.patch(
        'sasctl.core.Session._request_token_with_kerberos', return_value='token'
    ):
        s = Session('hostname')
        assert s.auth.access_token == 'token'

        s = Session('hostname', 'username')
        assert s.auth.access_token == 'token'

        s = Session('hostname', 'username@REALM')
        assert s.auth.access_token == 'token'


@mock.patch('sasctl.core.Session.get')
@mock.patch('sasctl.core.kerberos')
def test_request_token_with_kerberos_individual_calls(kerberos, get):
    """Test correct behavior of Session._request_token_with_kerberos()."""

    # Setup mock responses from kerberos package
    kerberos.GSS_C_MUTUAL_FLAG = 0
    kerberos.GSS_C_SEQUENCE_FLAG = 1
    kerberos.authGSSClientInit.return_value = (None, None)
    kerberos.authGSSClientUserName.return_value = f"{USERNAME}@THEFOREST"

    # Response to initial challenge
    response1 = mock.MagicMock()
    response1.status_code = 401
    response1.headers = {'www-authenticate': 'Negotiate'}

    # Response containing access token
    response2 = mock.MagicMock()
    response2.headers = {"Location": f"some stuff with an access_token={ACCESS_TOKEN}"}

    get.side_effect = [response1, response2]

    with mock.patch('sasctl.core.Session.__init__') as init:
        init.return_value = None
        s = Session()
        s._settings = {'domain': 'example.com', 'port': 443, 'protocol': 'https', 'username': None, 'password': None}
        s.verify = True

    token = s._request_token_with_kerberos()
    assert isinstance(token, OAuth2Token)
    assert token.access_token == ACCESS_TOKEN

    # Username should have been updated
    assert s.username == USERNAME


def test_request_token_with_kerberos_no_username():
    # no username, kerb available
    pass



def test_existing_token():
    """If an explicit token is provided it should be passed along."""

    s = Session('hostname', token=ACCESS_TOKEN)
    assert s.auth.access_token == ACCESS_TOKEN
    assert s.auth.refresh_token is None


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
            'sasctl.core.Session._get_authorization_token',
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
