#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import warnings

import betamax
from betamax_serializers import pretty_json
from betamax.cassette.cassette import Placeholder
import pytest
import six
from six.moves.urllib.parse import urlsplit
from sasctl import Session

from .matcher import RedactedPathMatcher

# Register 'mock' for easy importing in test modules
six.add_move(six.MovedModule('mock', 'mock', 'unittest.mock'))


def redact(interaction, cassette):
    """Remove sensitive or environment-specific information from cassettes
    before they are saved.

    Parameters
    ----------
    interaction
    cassette

    Returns
    -------

    """

    # Server name in Origin header may differ from hostname that was sent the
    # request.
    for origin in interaction.data['response']['headers'].get('Origin', []):
        host = urlsplit(origin).netloc
        if host != '' and Placeholder(placeholder='hostname.com', replace=host) not in cassette.placeholders:
            cassette.placeholders.append(Placeholder(placeholder='hostname.com', replace=host))

    def add_placeholder(pattern, string, placeholder, group):
        if isinstance(string, bytes):
            pattern = pattern.encode('utf-8')

        match = re.search(pattern, string)
        if match:
            old_text = match.group(group)
            cassette.placeholders.append(Placeholder(placeholder=placeholder, replace=old_text))

    add_placeholder(r"(?<=&password=)([^&]*)\b", interaction.data['request']['body']['string'], '*****', 1)
    add_placeholder('(?<=access_token":")[^"]*', interaction.data['response']['body']['string'], '[redacted]', 0)

    for index, header in enumerate(interaction.data['request']['headers'].get('Authorization', [])):
        # Betamax tries to replace Placeholders on all headers.  Mixed str/bytes headers will cause Betamax to break.
        if isinstance(header, bytes):
            header = header.decode('utf-8')
            interaction.data['request']['headers']['Authorization'][index] = header
        add_placeholder(r'(?<=Basic ).*', header, '[redacted]', 0)      # swat
        add_placeholder(r'(?<=Bearer ).*', header, '[redacted]', 0)     # sasctl


betamax.Betamax.register_serializer(pretty_json.PrettyJSONSerializer)
betamax.Betamax.register_request_matcher(RedactedPathMatcher)

from .matcher import PartialBodyMatcher
betamax.Betamax.register_request_matcher(PartialBodyMatcher)

with betamax.Betamax.configure() as config:
    config.cassette_library_dir = "tests/cassettes"
    config.default_cassette_options['record_mode'] = 'once'
    config.default_cassette_options['match_requests_on'] = ['method',
                                                            'redacted_path',
                                                            # 'partial_body',
                                                            'query']
    config.before_record(callback=redact)
    config.before_playback(callback=redact)


# Use the SASCTL_TEST_SERVERS variable to specify one or more servers that will
# be used for recording test cases
os.environ.setdefault('SASCTL_TEST_SERVERS', 'sasctl.example.com')
hostnames = [host.strip() for host in os.environ['SASCTL_TEST_SERVERS'].split(',')]

# Set dummy credentials if none were provided.
# Credentials don't matter if rerunning Betamax cassettes, but new recordings
# will fail.
os.environ.setdefault('SASCTL_SERVER_NAME', hostnames[0])
os.environ.setdefault('SASCTL_USER_NAME', 'dummyuser')
os.environ.setdefault('SASCTL_PASSWORD', 'dummypass')
os.environ.setdefault('SSLREQCERT', 'no')

with betamax.Betamax.configure() as config:
    for hostname in hostnames:
        config.define_cassette_placeholder('hostname.com', hostname)

    config.define_cassette_placeholder('hostname.com',
                                       os.environ['SASCTL_SERVER_NAME'])
    config.define_cassette_placeholder('USERNAME',
                                       os.environ['SASCTL_USER_NAME'])
    config.define_cassette_placeholder('*****',
                                       os.environ['SASCTL_PASSWORD'])


@pytest.fixture(scope='session', params=hostnames)
def credentials(request):
    auth = {'hostname': request.param,
            'username': os.environ['SASCTL_USER_NAME'],
            'password': os.environ['SASCTL_PASSWORD'],
            'verify_ssl': False
            }

    if 'SASCTL_AUTHINFO' in os.environ:
        auth['authinfo'] = os.path.expanduser(os.environ['SASCTL_AUTHINFO'])

    return auth


@pytest.fixture(scope='function')
def session(request, credentials):
    import warnings
    from six.moves import mock
    from betamax.fixtures.pytest import _casette_name
    from sasctl import current_session

    # Ignore FutureWarnings from betamax to avoid cluttering test results
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cassette_name = _casette_name(request, parametrized=False)

    # Need to instantiate a Session before starting Betamax recording,
    # but sasctl.Session makes requests (which should be recorded) during
    # __init__().  Mock __init__ to prevent from running and then manually
    # execute requests.Session.__init__() so Betamax can use the session.
    with mock.patch('sasctl.core.Session.__init__', return_value=None):
        recorded_session = Session()
        super(Session, recorded_session).__init__()

    with betamax.Betamax(recorded_session).use_cassette(cassette_name,
                                                        serialize_with='prettyjson') as recorder:
        recorder.start()

        # Manually run the sasctl.Session constructor.  Mock out calls to
        # underlying requests.Session.__init__ to prevent hooks placed by
        # Betamax from being reset.
        with mock.patch('sasctl.core.requests.Session.__init__'):
            recorded_session.__init__(**credentials)
            current_session(recorded_session)
        yield recorded_session
        recorder.stop()
        current_session(None)


@pytest.fixture
def missing_packages():
    """Creates a context manager that prevents the specified packages from being imported.

    Use the simulate packages missing during testing.

    Examples
    --------
    with missing_packages('os'):
        with pytest.raises(ImportError):
            import os

    """

    from six.moves import mock
    from contextlib import contextmanager

    @contextmanager
    def mocked_importer(packages):
        builtin_import = __import__

        # Accept single string or iterable of strings
        if isinstance(packages, str):
            packages = [packages]

        # Method that fails to load specified packages but otherwise behaves normally
        def _import(name, *args, **kwargs):
            if any(name == package for package in packages):
                raise ImportError()
            return builtin_import(name, *args, **kwargs)

        try:
            with mock.patch(six.moves.builtins.__name__ + '.__import__', side_effect=_import):
                yield
        finally:
            pass

    return mocked_importer


@pytest.fixture
def cas_session(request, credentials):
    import requests
    from betamax.fixtures.pytest import _casette_name
    from six.moves import mock
    swat = pytest.importorskip('swat')

    # Ignore FutureWarnings from betamax to avoid cluttering test results
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cassette_name = _casette_name(request, parametrized=False) + '_swat'


    # Must have an existing Session for Betamax to record
    recorded_session = requests.Session()

    with betamax.Betamax(recorded_session).use_cassette(cassette_name, serialize_with='prettyjson') as recorder:
        recorder.start()

        # CAS connection tries to create its own Session instance.
        # Inject the session being recorded into the CAS connection
        with mock.patch('swat.cas.rest.connection.requests.Session') as mocked:
            mocked.return_value = recorded_session
            with swat.CAS('https://{}/cas-shared-default-http/'.format(
                    credentials['hostname']),
                          username=credentials['username'],
                          password=credentials['password']) as s:

                # Strip out the session id from requests & responses.
                recorder.config.define_cassette_placeholder('[session id]', s._session)
                yield s
        recorder.stop()


@pytest.fixture
def astore(cas_session):
    pd = pytest.importorskip('pandas')
    datasets = pytest.importorskip('sklearn.datasets')

    ASTORE_NAME = 'astore'

    cas_session.loadactionset('decisionTree')

    raw = datasets.load_iris()
    iris = pd.DataFrame(raw.data, columns=raw.feature_names)
    iris = iris.join(pd.DataFrame(raw.target))
    iris.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

    tbl = cas_session.upload(iris).casTable
    r = tbl.decisiontree.gbtreetrain(target='Species',
                                       inputs=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
                                       ntree=10,
                                       savestate=ASTORE_NAME)
    return cas_session.CASTable(ASTORE_NAME)


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "incremental: tests should be executed in order and xfail if previous test fails.")


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" % previousfailed.name)


