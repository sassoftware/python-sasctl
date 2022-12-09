#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import os
import re
from contextlib import contextmanager
from unittest import mock
from urllib.parse import urlsplit

import betamax
import pytest
import requests
from betamax.cassette.cassette import Placeholder
from betamax_serializers import pretty_json

from sasctl import Session, current_session
from .betamax_utils import BinarySerializer, RedactedPathMatcher


# All version numbers for which we will attempt to find cassettes when replaying tests.
ALL_VIYA_VERSIONS = ["3.5", "2022.09"]


def get_cassette_file(request, version):
    """Generate the name and storage location of a cassette given the requesting test context.

    Parameters
    ----------
    request : pytest.FixtureRequest
    version : float or str

    Returns
    -------
    str, str
        relative path to cassette folder, name of cassette

    """
    test_type = request.node.path.parent.name
    test_set = request.node.path.with_suffix("").name
    test_class = request.node.cls.__name__ if request.node.cls else None
    test_name = request.node.originalname
    cassette_folder = f"tests/{test_type}/cassettes"

    if test_class:
        cassette_name = (
            f"{test_set}.{test_class}.{test_name}.viya_{str(version).replace('.', '')}"
        )
    else:
        cassette_name = f"{test_set}.{test_name}.viya_{str(version).replace('.', '')}"

    return cassette_folder, cassette_name


def redact(interaction, cassette):
    """Remove sensitive or environment-specific information from cassettes before they are saved.

    Parameters
    ----------
    interaction
    cassette

    Returns
    -------
    None

    """

    def add_placeholder(pattern, string, placeholder, group):
        """Use regex `pattern` to search `string` and replace any match with `placeholder`."""
        if isinstance(string, bytes):
            pattern = pattern.encode("utf-8")

        match = re.search(pattern, string)
        if match:
            old_text = match.group(group)
            cassette.placeholders.append(
                Placeholder(placeholder=placeholder, replace=old_text)
            )

    request = interaction.data["request"]
    response = interaction.data["response"]

    # Server name in Origin header may differ from hostname that was sent the
    # request.
    for origin in response["headers"].get("Origin", []):
        host = urlsplit(origin).netloc
        if (
            host != ""
            and Placeholder(placeholder="hostname.com", replace=host)
            not in cassette.placeholders
        ):
            cassette.placeholders.append(
                Placeholder(placeholder="hostname.com", replace=host)
            )

    # Redact the password
    if "string" in request["body"]:
        add_placeholder(
            r"(?<=&password=)([^&]*)\b",
            interaction.data["request"]["body"]["string"],
            "*****",
            1,
        )

    # If the response is from a login attempt then we need to redact the token details.
    if "string" in response["body"] and '"access_token":' in response["body"]["string"]:
        # Redact value of access token
        add_placeholder(
            '(?<="access_token":")[^"]*',
            response["body"]["string"],
            "[redacted]",
            0,
        )

        # Redact value of id token
        add_placeholder(
            '(?<="id_token":")[^"]*',
            response["body"]["string"],
            "[redacted]",
            0,
        )

        # Redact the names of the authorized scopes
        add_placeholder(
            '(?<="scope":")[^"]*',
            interaction.data["response"]["body"]["string"],
            "[redacted]",
            0,
        )

    for index, header in enumerate(request["headers"].get("Authorization", [])):
        # Betamax tries to replace Placeholders on all headers.  Mixed str/bytes headers will cause Betamax to break.
        if isinstance(header, bytes):
            header = header.decode("utf-8")
            request["headers"]["Authorization"][index] = header
        add_placeholder(r"(?<=Basic ).*", header, "[redacted]", 0)  # swat
        add_placeholder(r"(?<=Bearer ).*", header, "[redacted]", 0)  # sasctl


betamax.Betamax.register_serializer(pretty_json.PrettyJSONSerializer)
betamax.Betamax.register_serializer(BinarySerializer)
betamax.Betamax.register_request_matcher(RedactedPathMatcher)

# Replay cassettes only by default
# Can be overridden as necessary to update cassettes
# See https://betamax.readthedocs.io/en/latest/record_modes.html for details.
# NOTE: We've added a custom "live" record mode that bypasses all recording/replaying of cassettes
#       and allows test suite to be run against a live server.
record_mode = os.environ.get("SASCTL_RECORD_MODE", "none").lower()
if record_mode not in ("once", "new_episodes", "all", "none", "live"):
    record_mode = "none"

# Set a flag to indicate whether bypassing Betamax altogether.
if record_mode == "live":
    SKIP_REPLAY = True

    # Setting this back to a valid Betamax value to avoid downstream errors.
    record_mode = "once"
else:
    SKIP_REPLAY = False

# Use the SASCTL_TEST_SERVER variable to specify which server will be used for recording test cases
os.environ.setdefault("SASCTL_TEST_SERVER", "sasctl.example.com")

# Set dummy credentials if none were provided.
# Credentials don't matter if rerunning Betamax cassettes, but new recordings will fail.
os.environ.setdefault("SASCTL_SERVER_NAME", "sasctl.example.com")
os.environ.setdefault("SASCTL_USER_NAME", "dummyuser")
os.environ.setdefault("SASCTL_PASSWORD", "dummypass")
os.environ.setdefault("SSLREQCERT", "no")

# NOTE: SWAT gives CAS_CLIENT_SSL_CA_LIST precedence over SSLREQCERT, which will result in failed SSL verification
#       attempts unless CAS_CLIENT_SSL_CA_LIST is removed when bypassing SSL verification is desired.
if os.environ["SSLREQCERT"].lower() in ("no", "n", "false"):
    os.environ["CAS_CLIENT_SSL_CA_LIST"] = ""

# Configure Betamax
config = betamax.Betamax.configure()
# config.cassette_library_dir = 'tests/cassettes'
# config.default_cassette_options['serialize_with'] = 'prettyjson'
config.default_cassette_options["serialize_with"] = "binary"
config.default_cassette_options["record_mode"] = record_mode
config.default_cassette_options["match_requests_on"] = [
    "method",
    "redacted_path",
    # 'partial_body',
    "query",
]

# Create placeholder replacement values for any sensitive data that we know in advance.
config.define_cassette_placeholder("hostname.com", os.environ["SASCTL_TEST_SERVER"])
config.define_cassette_placeholder("hostname.com", os.environ["SASCTL_SERVER_NAME"])
config.define_cassette_placeholder("USERNAME", os.environ["SASCTL_USER_NAME"])
config.define_cassette_placeholder("*****", os.environ["SASCTL_PASSWORD"])

# Call redact() to remove sensitive data that isn't known in advance (like token values)
config.before_record(callback=redact)
# config.before_playback(callback=redact)

# We need to be able to run tests against a specific version of Viya when recording cassettes, but then run tests
# against all versions of Viya when replaying cassettes.  Use an environment variable during recording to track which
# version of Viya is being used, but use a list of known versions during test replay.
if record_mode in ("all", "once", "new_episodes"):
    viya_versions = os.getenv("SASCTL_SERVER_VERSION")

    if viya_versions is None:
        raise RuntimeError(
            "The SASCTL_SERVER_VERSION environment variable must be set when recording cassettes."
            "This variable should be set to the version number of the Viya environment to which you "
            "are connecting."
        )

    # Convert to a single-item list since pytest expects a list of values.
    viya_versions = [viya_versions]
elif record_mode == "none":
    # If replaying only, then try to test against each version
    viya_versions = ALL_VIYA_VERSIONS
else:
    # We're skipping Betamax record/replay altogether and running live, so this doesn't matter.
    viya_versions = []


@pytest.fixture(scope="session")
def credentials():
    auth = {
        "hostname": os.environ["SASCTL_TEST_SERVER"],
        "username": os.environ["SASCTL_USER_NAME"],
        "password": os.environ["SASCTL_PASSWORD"],
        "verify_ssl": False,
    }

    if "SASCTL_AUTHINFO" in os.environ:
        auth["authinfo"] = os.path.expanduser(os.environ["SASCTL_AUTHINFO"])

    return auth


@pytest.fixture(scope="function")
def session(request, credentials):
    # If we're bypassing Betamax altogether then just return the Session and we can avoid the mess of
    # setting up the cassette.
    if SKIP_REPLAY:
        yield Session(**credentials)
        current_session(None)
        return

    # # Check which version of Viya we are using to label the cassettes.
    # expected_version = os.getenv('SASCTL_SERVER_VERSION')
    # if expected_version is None:
    #     raise RuntimeError('The SASCTL_SERVER_VERSION environment variable must be set when recording cassettes.'
    #                        'This variable should be set to the version number of the Viya environment to which you '
    #                        'are connecting.')
    expected_version = request.param

    # Use the test information from pytest request instance to determine the name and folder location for the cassette.
    cassette_folder, cassette_name = get_cassette_file(request, expected_version)

    # Need to instantiate a Session before starting Betamax recording,
    # but sasctl.Session makes requests (which should be recorded) during
    # __init__().  Mock __init__ to prevent from running and then manually
    # execute requests.Session.__init__() so Betamax can use the session.
    with mock.patch("sasctl.core.Session.__init__", return_value=None):
        recorded_session = Session()
        super(Session, recorded_session).__init__()

    with betamax.Betamax(
        recorded_session, cassette_library_dir=cassette_folder
    ) as recorder:
        try:
            recorder.use_cassette(cassette_name)
        except ValueError:
            # If the requested cassette doesn't exist, Betamax will raise a ValueError.  If we are just replaying test
            # cases then we want to *try* to run tests against all versions of Viya.  However, don't fail if no test
            # has been recorded for the current Viya version - just skip the test and continue.
            if record_mode == "none":
                pytest.skip(f"No cassette found for version '{request.param}'")
            else:
                raise

        # Manually run the sasctl.Session constructor.  Mock out calls to
        # underlying requests.Session.__init__ to prevent hooks placed by
        # Betamax from being reset.
        with mock.patch("sasctl.core.requests.Session.__init__"):
            recorded_session.__init__(**credentials)
            current_session(recorded_session)

        # Verify that the Viya environment we're talking to is running the version of Viya that we expected.
        # This is a sanity check to ensure that we don't accidently record cassettes from version X labeled as
        # version Y.  Versions should be specified using version number (e.g. '3.5') for Viya 3 and release number
        # (e.g. '2022.01') for Viya 4.
        version = recorded_session.version_info()
        if (version < 4 and version != float(expected_version)) or (
            version >= 4 and version.release != expected_version
        ):
            raise RuntimeError(
                f"You are connected to a Viya environment with version {version} but are trying to "
                f"record cassettes labeled as version {expected_version}."
            )

        yield recorded_session
        current_session(None)


# betamax.cassette.interaction.Interaction  betamax.cassette.cassette.Cassette
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
            with mock.patch(builtins.__name__ + ".__import__", side_effect=_import):
                yield
        finally:
            pass

    return mocked_importer


@pytest.fixture(scope="function")
def cas_session(request, credentials):
    """

    Parameters
    ----------
    request : pytest.FixtureRequest
        Details of the test & associated parameters currently being executed by pytest.  Automatically passed by
        pytest framework.
    credentials : dict
        Credentials to use when establishing the CAS session.  Automatically passed by pytest framework since the
        `credentials` fixture is defined.

    Yields
    -------
    swat.CAS
        A CAS connection instance that is being recorded/replayed by Betamax.

    """
    swat = pytest.importorskip("swat")
    from swat.exceptions import SWATError

    # Bypass Betamax entirely if requested.
    if SKIP_REPLAY:
        with swat.CAS(
            "https://{}/cas-shared-default-http/".format(credentials["hostname"]),
            username=credentials["username"],
            password=credentials["password"],
        ) as s:
            yield s
        return

    # Use the test information from pytest request instance to determine the name and folder location for the cassette.
    cassette_folder, cassette_name = get_cassette_file(request, request.param)
    cassette_name += ".swat"

    # Must have an existing Session for Betamax to record
    recorded_session = requests.Session()

    with betamax.Betamax(
        recorded_session, cassette_library_dir=cassette_folder
    ) as recorder:
        try:
            recorder.use_cassette(cassette_name)
        except ValueError:
            # If the requested cassette doesn't exist, Betamax will raise a ValueError.  If we are just replaying test
            # cases then we want to *try* to run tests against all versions of Viya.  However, don't fail if no test
            # has been recorded for the current Viya version - just skip the test and continue.
            if record_mode == "none":
                pytest.skip(f"No cassette found for version '{request.param}'")
            else:
                raise

        # CAS connection tries to create its own Session instance.
        # Inject the session being recorded into the CAS connection
        with mock.patch("swat.cas.rest.connection.requests.Session") as mocked:
            mocked.return_value = recorded_session
            s = None
            try:
                s = swat.CAS(
                    "https://{}/cas-shared-default-http/".format(
                        credentials["hostname"]
                    ),
                    username=credentials["username"],
                    password=credentials["password"],
                )

                # Strip out the session id from requests & responses.
                recorder.config.define_cassette_placeholder("[session id]", s._session)
                yield s
            finally:
                try:
                    if hasattr(s, "close"):
                        s.close()
                except SWATError:
                    # session was closed during testing
                    pass


@pytest.fixture
def iris_astore(cas_session):
    pd = pytest.importorskip("pandas")
    datasets = pytest.importorskip("sklearn.datasets")

    ASTORE_NAME = "astore"

    cas_session.loadactionset("decisionTree")

    raw = datasets.load_iris()
    iris = pd.DataFrame(raw.data, columns=raw.feature_names)
    iris = iris.join(pd.DataFrame(raw.target))
    iris.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]

    tbl = cas_session.upload(iris).casTable
    _ = tbl.decisiontree.gbtreetrain(
        target="Species",
        inputs=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"],
        nominal=["Species"],
        ntree=10,
        savestate=ASTORE_NAME,
    )
    return cas_session.CASTable(ASTORE_NAME)


@pytest.fixture
def airline_dataset():
    """Sentiment analysis dataset."""
    pd = pytest.importorskip("pandas")

    df = pd.read_csv("examples/data/airline_tweets.csv")
    df = df[
        [
            "airline_sentiment",
            "airline",
            "name",
            "tweet_location",
            "tweet_id",
            "tweet_created",
            "retweet_count",
            "text",
        ]
    ]
    return df


@pytest.fixture
def boston_dataset():
    """Regression dataset."""
    pd = pytest.importorskip("pandas")

    df = pd.read_csv("examples/data/boston_house_prices.csv")

    # Uppercase column names to match names used by scikit-learn (dataset was originally loaded through
    # sklearn before it was removed in v1.2).
    df.columns = [c.upper() for c in df.columns]
    df = df.rename(columns={"MEDV": "Price"})
    return df


@pytest.fixture
def cancer_dataset():
    """Binary classification dataset."""
    pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    from sklearn import datasets

    raw = datasets.load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df["Type"] = raw.target
    df.Type = df.Type.astype("category")
    df.Type.cat.categories = raw.target_names
    return df


@pytest.fixture
def iris_dataset():
    """Multi-class classification dataset."""
    pd = pytest.importorskip("pandas")

    df = pd.read_csv("examples/data/iris.csv")
    df.Species = df.Species.astype("category")
    return df


@pytest.fixture
def cache(request):
    """Wraps the built-in py.test cache with a custom cache that segregates data based on test grouping."""
    return Cache(request)
    # return object with get/set


class Cache:
    """Test grouping-aware cache object.

    Simple wrapper around the py.test cache object but it ensures that any data written by a test with an assigned
    grouping (Viya version) is only read by tests with the same grouping.  This means that a test function can
    write to the cache using the same key (e.g. "MY_CACHED_DATA") and each version of that test function will cache
    it's data separately.

    """

    def __init__(self, request):
        self.__request = request

    @property
    def grouping(self):
        return getattr(self.__request.node, "grouping", "")

    def get(self, key, default):
        key = self._format_key(key)
        return self.__request.config.cache.get(key, default)

    def set(self, key, value):
        key = self._format_key(key)
        return self.__request.config.cache.set(key, value)

    def _format_key(self, key):
        if self.grouping:
            return f"{self.grouping}/{key}"
        return key


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "incremental: tests should be executed in order and xfail if previous test fails.",
    )


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            # Create a dictionary to track which version(s) of the test failed
            if not hasattr(parent, "previousfailed"):
                parent._previousfailed = {}

            # The id of the test is deteremined by its parameterization.  We just want to know if the test was
            # for Viya 3.5 or 2020.01, 2022.09, etc.  Try to check the parameter assigned to known fixtures like
            # `session`.  If that fails, we'll just use the id generated by pytest.
            if "session" in item.callspec.params:
                key = item.callspec.params["session"]
            elif "cas_session" in item.callspec.params:
                key = item.callspec.params["cas_session"]
            else:
                key = item.callspec.id

            # Track that this test was the last test to fail for this Viya version
            parent._previousfailed[key] = item


def pytest_runtest_setup(item):
    # We need a way to identify which version of Viya each individual test targets.  This lets us ensure that cached
    # data doesn't get mixed across different versions of the same test function, or that we don't xfail an
    # incremental test because of a previous failure by a test associated with a different Viya version.
    # The `id` of each test is generated by py.test based on the test parameterization and may not match across all
    # test cases with the same Viya version, so we need an alternative method.  Instead, we use the parameter passed
    # to `session` or `cas_session` and only fall back to `id` if neither fixture was used.
    if hasattr(item, "callspec"):
        if "session" in item.callspec.params:
            item.grouping = item.callspec.params["session"]
        elif "cas_session" in item.callspec.params:
            item.grouping = item.callspec.params["cas_session"]
        else:
            item.grouping = item.callspec.id

    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            # If a previous test for the same Viya version has failed then we can just skip this test.
            if item.grouping in previousfailed:
                pytest.xfail(
                    f"previous test failed {previousfailed[item.grouping].name}"
                )


def pytest_generate_tests(metafunc):
    """Set or change the parameters passed to each test function.

    Automatically called by pytest framework during test collection before any tests are execution.  Called once for
    each test case and provides an opportunity to set or change the parameters passed into the test case.

    The `session` and `cas_session` test fixtures must be parameterized with version of Viya being used in order for
    Betamax to include the version number in the cassette name & create different cassettes for different versions.
    However, if the fixtures are parameterized independently, pytest will generate the cartesian product of the
    parameter lists for any test that uses both `session` and `cas_session` fixtures.  This results in nonsensical
    tests like 3.5-4.0 and 4.0-3.5 which would indicate the test is using two servers with different Viya versions.

    Instead, we want to explicitly provide the combinations of parameters for both fixtures to ensure that the
    fixtures only use combinations with identical version numbers (i.e. `session` and `cas_session` both receive
    the '3.5' parameter at the same time).

    """

    # We need to provide parameters for one or both of `session` and `cas_session` if they're being used by the test.
    fixtures_to_parameterize = [
        f for f in ("session", "cas_session") if f in metafunc.fixturenames
    ]

    # Build a list of combinations that will be used to parameterize the test.
    # Example: [('3.5', '3.5'), ('2022.01', '2022.01'), ('2022.02', '2022.02')]
    params = [[v] * len(fixtures_to_parameterize) for v in viya_versions]

    # Instruct pytest to use the list of parameter combinations.  Indirect=True tells pytest that the parameter values
    # (the version numbers) should not be passed directly to the test function as parameter values.  Instead, they
    # should be passed to the fixtures (`session` and `cas_session`) which will use them to generate the values that
    # are provided to the test function parameters
    if fixtures_to_parameterize:
        metafunc.parametrize(fixtures_to_parameterize, params, indirect=True)
