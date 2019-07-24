#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os

import pytest
from six.moves import mock

from sasctl import Session, current_session


def test_new_session(missing_packages):
    HOST = 'example.com'
    USERNAME = 'user'
    PASSWORD = 'password'

    # Ensure no dependency on swat required
    with missing_packages('swat'):
        with mock.patch('sasctl.core.Session.get_token'):
            s = Session(HOST, USERNAME, PASSWORD)
        assert USERNAME == s.username
        assert HOST == s._settings['domain']
        assert 'https' == s._settings['protocol']
        assert USERNAME == s._settings['username']
        assert PASSWORD == s._settings['password']

    # Tests don't reset global variables (_session) so explicitly cleanup
    current_session(None)


def test_current_session():
    assert current_session() is None

    # Initial session should automatically become the default
    with mock.patch('sasctl.core.Session.get_token'):
        s = Session('example.com', 'user', 'password')
    assert current_session() == s

    # Subsequent sessions should not overwrite the default
    with mock.patch('sasctl.core.Session.get_token'):
        s2 = Session('example.com', 'user2', 'password')
    assert current_session() != s2
    assert current_session() == s

    # Explicitly set new current session
    with mock.patch('sasctl.core.Session.get_token'):
        s3 = current_session('example.com', 'user3', 'password')
    assert current_session() == s3

    # Explicitly change current session
    with mock.patch('sasctl.core.Session.get_token'):
        s4 = Session('example.com', 'user4', 'password')
    current_session(s4)
    assert 'user4' == current_session().username

    with mock.patch('sasctl.core.Session.get_token'):
        with Session('example.com', 'user5', 'password') as s5:
            with Session('example.com', 'user6', 'password') as s6:
                assert current_session() == s6
                assert current_session() != s5
                assert current_session().username == 'user6'
            assert current_session().username == 'user5'
        assert current_session().username == 'user4'


def test_swat_connection_reuse():
    import base64
    swat = pytest.importorskip('swat')

    HOST = 'example.com'
    PORT = 8777
    PROTOCOL = 'https'
    USERNAME = 'username'
    PASSWORD = 'password'

    mock_cas = mock.Mock(spec=swat.CAS)
    mock_cas._hostname = 'casserver.com'
    mock_cas._sw_connection = mock.Mock(
        spec=swat.cas.rest.connection.REST_CASConnection)
    mock_cas._sw_connection._auth = base64.b64encode(
        (USERNAME + ':' + PASSWORD).encode())
    mock_cas.get_action.return_value = lambda: swat.cas.results.CASResults(
        port=PORT,
        protocol=PROTOCOL,
        restPrefix='/cas-shared-default-http',
        virtualHost=HOST)
    with mock.patch('sasctl.core.Session.get_token'):
        with Session(mock_cas) as s:
            # Should reuse port # from SWAT connection
            # Should query CAS to find the HTTP connection _settings
            assert HOST == s._settings['domain']
            assert PORT == s._settings['port']
            assert PROTOCOL == s._settings['protocol']
            assert USERNAME == s._settings['username']
            assert PASSWORD == s._settings['password']
            assert '{}://{}:{}/test'.format(PROTOCOL, HOST,
                                            PORT) == s._build_url('/test')

        with Session(HOST, username=USERNAME, password=PASSWORD, protocol=PROTOCOL,
                     port=PORT) as s:
            assert HOST == s._settings['domain']
            assert PORT == s._settings['port']
            assert PROTOCOL == s._settings['protocol']
            assert USERNAME == s._settings['username']
            assert PASSWORD == s._settings['password']
            assert '{}://{}:{}/test'.format(PROTOCOL, HOST,
                                            PORT) == s._build_url('/test')

        with Session(HOST, username=USERNAME, password=PASSWORD,
                     protocol=PROTOCOL) as s:
            assert HOST == s._settings['domain']
            assert s._settings[
                       'port'] is None  # Let Requests determine default port
            assert PROTOCOL == s._settings['protocol']
            assert USERNAME == s._settings['username']
            assert PASSWORD == s._settings['password']
            assert '{}://{}/test'.format(PROTOCOL, HOST) == s._build_url(
                '/test')


def test_log_filtering(caplog):
    caplog.set_level(logging.DEBUG, logger='sasctl.core.session')

    HOST = 'example.com'
    USERNAME = 'secretuser'
    PASSWORD = 'secretpassword'
    ACCESS_TOKEN = 'secretaccesstoken'
    REFRESH_TOKEN = 'secretrefreshtoken'
    CLIENT_SECRET = 'clientpassword'
    CONSUL_TOKEN = 'supersecretconsultoken!'

    sensitive_data = [PASSWORD, ACCESS_TOKEN, REFRESH_TOKEN, CLIENT_SECRET,
                      CONSUL_TOKEN]

    with mock.patch('requests.Session.send') as mocked:
        # Response to every request with a response that contains sensitive data
        # Access token should also be used to set session.auth
        mocked.return_value.status_code = 200
        mocked.return_value.raise_for_status.return_value = None
        mocked.return_value.json.return_value = {
            'access_token': ACCESS_TOKEN,
            'refresh_token': REFRESH_TOKEN
        }
        mocked.return_value.url = 'http://' + HOST
        mocked.return_value.headers = {}
        mocked.return_value.body = json.dumps(
            mocked.return_value.json.return_value)
        mocked.return_value._content = mocked.return_value.body

        with Session(HOST, USERNAME, PASSWORD) as s:
            assert s.auth is not None
            assert mocked.return_value == s.get('/fakeurl')
            assert mocked.return_value == s.post('/fakeurl',
                                                 headers={
                                                     'X-Consul-Token': CONSUL_TOKEN},
                                                 json={
                                                     'client_id': 'TestClient',
                                                     'client_secret': CLIENT_SECRET})

            # Correct token should have been set
            assert 'secretaccesstoken' == s.auth.token

            # No sensitive information should be contained in the log records
            assert len(caplog.records) > 0
            for r in caplog.records:
                for d in sensitive_data:
                    assert d not in r.message


def test_ssl_context():
    import os
    from sasctl.core import SSLContextAdapter

    # Cleanup any env vars currently set
    if 'CAS_CLIENT_SSL_CA_LIST' in os.environ: del os.environ['CAS_CLIENT_SSL_CA_LIST']
    if 'REQUESTS_CA_BUNDLE' in os.environ: del os.environ['REQUESTS_CA_BUNDLE']
    if 'SSLREQCERT' in os.environ: del os.environ['SSLREQCERT']

    # Should default to SSLContextAdapter if no certificate paths are set
    with mock.patch('sasctl.core.Session.get_token', return_value='token'):
        s = Session('hostname', 'username', 'password')
    assert isinstance(s.get_adapter('https://'), SSLContextAdapter)

    # If only the Requests env variable is set, it should be used
    os.environ['REQUESTS_CA_BUNDLE'] = 'path_for_requests'
    with mock.patch('sasctl.core.Session.get_token', return_value='token'):
        s = Session('hostname', 'username', 'password')
    assert 'CAS_CLIENT_SSL_CA_LIST' not in os.environ
    assert not isinstance(s.get_adapter('https://'), SSLContextAdapter)

    # If SWAT env variable is set, it should override the Requests variable
    os.environ['CAS_CLIENT_SSL_CA_LIST'] = 'path_for_swat'
    with mock.patch('sasctl.core.Session.get_token', return_value='token'):
        s = Session('hostname', 'username', 'password')
    assert os.environ['CAS_CLIENT_SSL_CA_LIST'] == os.environ['REQUESTS_CA_BUNDLE']
    assert not isinstance(s.get_adapter('https://'), SSLContextAdapter)

    # Cleanup
    del os.environ['CAS_CLIENT_SSL_CA_LIST']
    del os.environ['REQUESTS_CA_BUNDLE']

    # If SWAT env variable is set, it should override the Requests variable
    os.environ['SSLCALISTLOC'] = 'path_for_swat'
    with mock.patch('sasctl.core.Session.get_token', return_value='token'):
        s = Session('hostname', 'username', 'password')
    assert os.environ['SSLCALISTLOC'] == os.environ['REQUESTS_CA_BUNDLE']
    assert 'CAS_CLIENT_SSL_CA_LIST' not in os.environ
    assert not isinstance(s.get_adapter('https://'), SSLContextAdapter)

    # Cleanup
    del os.environ['SSLCALISTLOC']
    del os.environ['REQUESTS_CA_BUNDLE']


def test_verify_ssl():
    with mock.patch('sasctl.core.Session.get_token', return_value='token'):
        # Should verify SSL by default
        s = Session('hostname', 'username', 'password')
        assert s.verify == True

        # Specify true with no issues
        s = Session('hostname', 'username', 'password', verify_ssl=True)
        assert s.verify == True

        # Explicitly disable SSL verification
        s = Session('hostname', 'username', 'password', verify_ssl=False)
        assert s.verify == False

        # Reuse SWAT env variable, if specified
        os.environ['SSLREQCERT'] = 'NO'
        s = Session('hostname', 'username', 'password')
        assert s.verify == False

        os.environ['SSLREQCERT'] = 'no'
        s = Session('hostname', 'username', 'password')
        assert s.verify == False

        os.environ['SSLREQCERT'] = 'false'
        s = Session('hostname', 'username', 'password')
        assert s.verify == False

        # Explicit should take precedence over environment variables
        s = Session('hostname', 'username', 'password', verify_ssl=True)
        assert s.verify == True


def test_kerberos():
    with mock.patch('sasctl.core.Session._get_token_with_kerberos',
                    return_value='token'):
        s = Session('hostname')
        assert s.auth.token == 'token'

        s = Session('hostname', 'username')
        assert s.auth.token == 'token'

        s = Session('hostname', 'username@REALM')
        assert s.auth.token == 'token'

def test_authentication_failure():
    from sasctl.core import AuthenticationError

    with mock.patch('sasctl.core.Session.request') as request:
        request.return_value.status_code = 401

        with pytest.raises(AuthenticationError):
            Session('hostname', 'username', 'password')

