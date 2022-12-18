#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import copy
import logging
import json
import netrc
import os
import re
import ssl
import warnings
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import requests
import requests.exceptions
import yaml
from packaging import version
from requests.adapters import HTTPAdapter
from urllib.parse import urlsplit, urlunsplit
from urllib.error import HTTPError

try:
    import swat
except ImportError:
    swat = None

try:
    import kerberos
except ImportError:
    try:
        import winkerberos as kerberos
    except ImportError:
        kerberos = None

from .utils.cli import sasctl_command
from .utils.misc import versionadded
from . import exceptions

logger = logging.getLogger(__name__)

_session = None


def _pformat(text):
    from pprint import pformat

    try:
        return pformat(json.loads(text))
    except (TypeError, UnicodeDecodeError, ValueError):
        try:
            return pformat(text)
        except UnicodeDecodeError:
            return text


def _redact(pattern, repl, string):
    is_bytes = isinstance(string, bytes)

    try:
        string = string.decode("utf-8") if is_bytes else string
        string = re.sub(pattern, repl, string)
        string = string.encode("utf-8") if is_bytes else string
    except UnicodeDecodeError:
        pass
    return string


def _filter_password(r):
    if hasattr(r, "body") and r.body is not None:
        # Filter password from 'grant_type=password&username=<user>&password=<password>' during Post to Logon service.
        r.body = _redact(r"(?<=&password=)([^&]*)\b", "*****", r.body)

        # Filter client secret {"client_secret": "<password>"}
        r.body = _redact('(?<=client_secret": ")[^"]*', "*****", r.body)
    return r


def _filter_token(r):
    # Redact "Bearer <token>" in Authorization headers
    if hasattr(r, "headers") and "Authorization" in r.headers:
        r.headers["Authorization"] = _redact(
            r"(?<=Bearer ).*", "[redacted]", r.headers["Authorization"]
        )

    # Redact "Basic <base64 encoded pw> in Authorization headers.  This covers client ids & client secrets.
    if hasattr(r, "headers") and "Authorization" in r.headers:
        r.headers["Authorization"] = _redact(
            r"(?<=Basic ).*", "[redacted]", r.headers["Authorization"]
        )

    # Redact Consul token from headers.  Should only appear when registering a new client
    if hasattr(r, "headers") and "X-Consul-Token" in r.headers:
        r.headers["X-Consul-Token"] = "[redacted]"

    # Redact "access_token":"<token>" in response from SASLogon service
    if hasattr(r, "_content"):
        r._content = _redact('(?<=access_token":")[^"]*', "[redacted]", r._content)

    return r


DEFAULT_FILTERS = [_filter_password, _filter_token]


def current_session(*args, **kwargs):
    """Gets and sets the current session.

    If call with no arguments, the current session instance is returned, or
    None if no session has been created yet.  If called with an existing session
    instance, that session will be set as the default.  Otherwise, a new
    `Session` is instantiated using the provided arguments and set as the
    default.

    Parameters
    ----------
    args : any
    kwargs : any

    Returns
    -------
    Session

    Examples
    --------

    Get the current session

    >>> current_session()
    <sasctl.core.Session object at 0x1393fc550>

    Clear the current session

    >>> current_session(None)


    Make a new session current

    >>> current_session('example.com', 'knight', 'Ni!')
    <sasctl.core.Session object at 0x15a9df491>

    """
    global _session  # skipcq PYL-W0603

    # Explicitly set or clear the current session
    if len(args) == 1 and (isinstance(args[0], Session) or args[0] is None):
        _session = args[0]
    # Create a new session
    elif args:
        _session = Session(*args, **kwargs)

    return _session


class OAuth2Token(requests.auth.AuthBase):
    def __init__(
        self,
        access_token,
        refresh_token=None,
        expiration=None,
        expires_in=None,
        **kwargs,
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expiration = expiration

        if expires_in is not None:
            self.expiration = datetime.now() + timedelta(seconds=expires_in)

    def __call__(self, r):
        r.headers["Authorization"] = "Bearer " + self.access_token
        return r

    @property
    def is_expired(self):
        if self.expiration is None:
            return False

        return self.expiration < datetime.now()


class RestObj(dict):
    def __getattr__(self, item):
        # Only called when __getattribute__ failed to find the attribute
        # Return the item from underlying dictionary if possible.
        if item in self:
            result = self[item]

            if isinstance(result, dict):
                return RestObj(result)

            return result

        raise AttributeError(
            "'%s' object has no attribute '%s'" % (self.__class__.__name__, item)
        )

    def __repr__(self):
        headers = getattr(self, "_headers", {})

        return "%s(headers=%r, data=%s)" % (
            self.__class__,
            headers,
            super(RestObj, self).__repr__(),
        )

    def __str__(self):
        if "name" in self:
            return str(self["name"])
        if "id" in self:
            return str(self["id"])
        return repr(self)


class SSLContextAdapter(HTTPAdapter):
    """HTTPAdapter that uses the default SSL context on the machine."""

    def __init__(self, *args, **kwargs):
        self.assert_hostname = kwargs.pop("assert_hostname", True)
        requests.adapters.HTTPAdapter.__init__(self, *args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.check_hostname = self.assert_hostname
        kwargs["ssl_context"] = context
        kwargs["assert_hostname"] = self.assert_hostname
        return super(SSLContextAdapter, self).init_poolmanager(*args, **kwargs)


class Session(requests.Session):
    """Establish a connection to a SAS Viya server.

    Parameters
    ----------
    hostname : str or swat.CAS
        Name of the server to connect to or an established swat.CAS session.
    username : str, optional
        Username for authentication.  Not required if `host` is a CAS
        connection, if Kerberos is used, or if `token` is provided.  If using Kerberos and an explicit
        username is desired, maybe be a string in 'user@REALM' format.
    password : str, optional
        Password for authentication.  Not required when `host` is a CAS
        connection, `authinfo` is provided, `token` is provided, or Kerberos is used.
    authinfo : str, optional
        Path to a .authinfo or .netrc file from which credentials should be
        pulled.
    protocol : {'http', 'https'}
        Whether to use HTTP or HTTPS connections.  Defaults to `https`.
    port : int, optional
        Port number for the connection if a non-standard port is used.
        Defaults to 80 or 443 depending on `protocol`.
    verify_ssl : bool, optional
        Whether server-side SSL certificates should be verified.  Defaults
        to true.  Ignored for HTTP connections.
    token : str, optional
        OAuth token to use for authorization.
    client_id : str, optional
        Client ID requesting access.  Use if connection to Viya should be
        made using `client_credentials` method.
    client_secret : str, optional
        Client secret for client requesting access. Required if `client_id`
        is provided.


    Attributes
    ----------
    message_log : logging.Logger
        A log to which all REST request and response messages will be sent.  Attach a handler using
        `add_logger()` to capture these messages.

    filters : list of callable
        A collection of functions that will be called with each request and response object *prior* to logging the
        messages, allowing any sensitive information to be removed first.

    """

    PROFILE_PATH = "~/.sas/viya-api-profiles.yaml"

    def __init__(
        self,
        hostname,
        username=None,
        password=None,
        authinfo=None,
        protocol=None,
        port=None,
        verify_ssl=None,
        token=None,
        client_id=None,
        client_secret=None,
        consul_token=None,
    ):
        super(Session, self).__init__()

        # Determine whether or not server SSL certificates should be verified.
        if verify_ssl is None:
            verify_ssl = os.environ.get("SSLREQCERT", "yes")
            verify_ssl = str(verify_ssl).lower() not in ("no", "false")

        self._id = uuid4().hex
        self.message_log = logger.getChild("session.%s" % self._id)

        # If certificate path has already been set for SWAT package, make
        # Requests module reuse it.
        for k in ["SSLCALISTLOC", "CAS_CLIENT_SSL_CA_LIST"]:
            if k in os.environ:
                os.environ["REQUESTS_CA_BUNDLE"] = os.environ[k]
                break

        # If certificate path hasn't been specified in either environment
        # variable, replace the default adapter with one that will use the
        # machine's default SSL _settings.
        if "REQUESTS_CA_BUNDLE" not in os.environ:
            if verify_ssl:
                # Skip hostname verification if IP address specified instead
                # of DNS name.  Prevents error from urllib3.
                try:
                    from urllib3.util.ssl_ import is_ipaddress
                except ImportError:
                    # is_ipaddres not present in older versions of urllib3
                    def is_ipaddress(hst):
                        return re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", hst)

                verify_hostname = not is_ipaddress(hostname)
                adapter = SSLContextAdapter(assert_hostname=verify_hostname)

                self.mount("https://", adapter)

        # If we're skipping SSL verification, urllib3 will raise InsecureRequestWarnings on
        # every request.  Insert a warning filter so these warnings only appear on the first request.
        if not verify_ssl:
            from urllib3.exceptions import InsecureRequestWarning

            warnings.simplefilter("default", InsecureRequestWarning)

        self.filters = DEFAULT_FILTERS

        # Reuse an existing CAS connection if possible
        if swat and isinstance(hostname, swat.CAS):
            if isinstance(
                hostname._sw_connection, swat.cas.rest.connection.REST_CASConnection
            ):
                import base64

                # Use the httpAddress action to retieve information about
                # REST endpoints
                httpAddress = hostname.get_action("builtins.httpAddress")
                address = httpAddress()
                domain = address.virtualHost
                # httpAddress action may return virtualHost = ''
                # if this happens, try the CAS host
                if not domain:
                    domain = hostname._sw_connection._current_hostname
                protocol = address.protocol
                port = address.port
                auth = hostname._sw_connection._auth.decode("utf-8").replace(
                    "Basic ", ""
                )
                username, password = base64.b64decode(auth).decode("utf-8").split(":")
            else:
                raise ValueError(
                    "A 'swat.CAS' session can only be reused "
                    "when it's connected via the REST APIs."
                )
        else:
            url = urlsplit(hostname)

            # Extract http/https from domain name if present and protocol not explicitly given
            protocol = protocol or url.scheme
            domain = url.hostname or str(hostname)

        self._settings = {
            "protocol": protocol or "https",
            "domain": domain,
            "port": port,
            "username": username,
            "password": password,
        }

        if password is None and client_secret is None:
            # Try to get credentials from .authinfo or .netrc files.
            # If no file path was specified, the default locations will
            # be checked.
            try:
                auth = swat.utils.authinfo.query_authinfo(
                    domain, user=username, path=authinfo
                )
                if auth:
                    self._settings["username"] = auth.get("user")
                    self._settings["password"] = auth.get("password")
            except AttributeError:
                # If swat package or authinfo module not available
                pass

            # Not able to load credentials using SWAT.  Try Netrc.
            if self._settings["password"] is None:
                try:
                    parser = netrc.netrc(authinfo)
                    values = parser.authenticators(domain)
                    if values:
                        found_user, _, found_pass = values

                        # Only use the credentials if they match the specified user (if
                        # one was specified).
                        if username is None or username.lower() == found_user:
                            self._settings["username"] = found_user
                            self._settings["password"] = found_pass
                except (OSError, IOError):
                    pass  # netrc throws if $HOME is not set

        # Set this prior authentication attempts
        self.verify = verify_ssl

        if consul_token:
            self.auth = self._request_token_with_consul(
                consul_token, client_id=client_id
            )
        else:
            # Find a suitable authentication mechanism and build an auth header
            self.auth = self._get_authorization_token(
                token=token,
                username=self.username,
                password=self._settings["password"],
                client_id=client_id,
                client_secret=client_secret,
            )

        # Used to cache version info
        self._version_info = None

        # Used for context manager
        self._old_session = current_session()
        current_session(self)

    def add_logger(self, handler, level=None):
        """Log session requests and responses.

        Parameters
        ----------
        handler : logging.Handler
            A Handler instance to use for logging the requests and responses.
        level : int, optional
            The logging level to assign to the handler.  Ignored if handler's
            logging level is already set.  Defaults to DEBUG.

        Returns
        -------
        handler


        .. versionadded:: 1.2.0

        """
        level = level or logging.DEBUG

        if handler.level == logging.NOTSET:
            handler.setLevel(level)

        self.message_log.addHandler(handler)

        if self.message_log.level == logging.NOTSET:
            self.message_log.setLevel(handler.level)

        return handler

    def add_stderr_logger(self, level=None):
        """Log session requests and responses to stderr.

        Parameters
        ----------
        level : int, optional
            The logging level of the handler.  Defaults to logging.DEBUG

        Returns
        -------
        logging.StreamHandler

        """
        return self.add_logger(logging.StreamHandler(), level)

    @versionadded(version="1.5.4")
    def as_swat(self, server=None, **kwargs):
        """Create a SWAT connection to a CAS server.

        Uses the authentication information from the session to establish a CAS connection using SWAT.

        Parameters
        ----------
        server : str, optional
            The logical name of the CAS server, not the hostname.  Defaults to "cas-shared-default".
        kwargs : any
            Additional arguments to pass to the `swat.CAS` constructor.  Can be used to override this method's
            default behavior or customize the CAS session.

        Returns
        -------
        swat.CAS
            An active SWAT connection

        Raises
        ------
        RuntimeError
            If `swat` package is not available.

        Examples
        --------
        >>> sess = Session('example.sas.com')
        >>> with sess.as_swat() as conn:
        ...    conn.listnodes()
        CASResults([('nodelist', Node List
                      name        role connected   IP Address
        0  example.sas.com  controller       Yes  127.0.0.1)])

        """
        server = server or "cas-shared-default"

        if swat is None:
            raise RuntimeError(
                "The 'swat' package must be installed to create a SWAT connection."
            )

        # Construct the CAS server's URL
        url = "{}://{}/{}-http/".format(
            self._settings["protocol"], self.hostname, server
        )

        kwargs.setdefault("hostname", url)

        # Starting in SWAT v1.8 oauth tokens could be passed directly in the password param.
        # Otherwise, use the username & password to re-authenticate.
        # Use this sessions info to connect to CAS unless user has explicitly give a value (even if None)
        if version.parse(swat.__version__) >= version.parse("1.8"):
            kwargs.setdefault("username", None)
            kwargs.setdefault("password", self.auth.access_token)
        else:
            kwargs.setdefault("username", self.username)
            kwargs.setdefault("password", self._settings["password"])

        orig_sslreqcert = os.environ.get("SSLREQCERT")

        # If SSL connections to microservices are not being verified, don't attempt
        # to verify connections to CAS - most likely certs are not in place.
        if not self.verify:
            os.environ["SSLREQCERT"] = "no"

        try:
            cas = swat.CAS(**kwargs)
            cas.setsessopt(messagelevel="warning")
        finally:
            # Reset environment variable to whatever it's original value was
            if orig_sslreqcert:
                os.environ["SSLREQCERT"] = orig_sslreqcert

        return cas

    @property
    def username(self):
        return self._settings.get("username")

    @property
    def hostname(self):
        return self._settings.get("domain")

    def send(self, request, **kwargs):
        if self.message_log.isEnabledFor(logging.DEBUG):
            r = copy.deepcopy(request)
            for filter in self.filters:
                r = filter(r)

            self.message_log.debug(
                "HTTP/1.1 {verb} {url}\n{headers}\nBody:\n{body}".format(
                    verb=r.method,
                    url=r.url,
                    headers="\n".join(
                        "{}: {}".format(k, v) for k, v in r.headers.items()
                    ),
                    body=_pformat(r.body),
                )
            )
        else:
            self.message_log.info("HTTP/1.1 %s %s", request.method, request.url)

        response = super(Session, self).send(request, **kwargs)

        if self.message_log.isEnabledFor(logging.DEBUG):
            r = copy.deepcopy(response)
            for filter in self.filters:
                r = filter(r)

            self.message_log.debug(
                "HTTP {status} {url}\n{headers}\nBody:\n{body}".format(
                    status=r.status_code,
                    url=r.url,
                    headers="\n".join(
                        "{}: {}".format(k, v) for k, v in r.headers.items()
                    ),
                    body=_pformat(r.text),
                )
            )
        else:
            self.message_log.info("HTTP/1.1 %s %s", response.status_code, response.url)

        return response

    def request(
        self,
        method,
        url,
        params=None,
        data=None,
        headers=None,
        cookies=None,
        files=None,
        auth=None,
        timeout=None,
        allow_redirects=True,
        proxies=None,
        hooks=None,
        stream=None,
        verify=None,
        cert=None,
        json=None,
    ):

        url = self._build_url(url)
        verify = verify or self.verify

        try:
            r = super(Session, self).request(
                method,
                url,
                params,
                data,
                headers,
                cookies,
                files,
                auth,
                timeout,
                allow_redirects,
                proxies,
                hooks,
                stream,
                verify,
                cert,
                json,
            )

            if r.status_code == 401:
                auth_header = r.headers.get("WWW-Authenticate", "").lower()

                # Access token expired, need to refresh it (if we can)
                if "access token expired" in auth_header:
                    try:
                        self.auth = self._request_token_with_oauth(
                            refresh_token=self.auth.refresh_token
                        )

                        # Repeat the request
                        r = super(Session, self).request(
                            method,
                            url,
                            params,
                            data,
                            headers,
                            cookies,
                            files,
                            auth,
                            timeout,
                            allow_redirects,
                            proxies,
                            hooks,
                            stream,
                            verify,
                            cert,
                            json,
                        )
                    except exceptions.AuthorizationError:
                        pass

            return r
        except requests.exceptions.SSLError as e:
            if "REQUESTS_CA_BUNDLE" not in os.environ:
                raise RuntimeError(
                    "SSL handshake failed.  The 'REQUESTS_CA_BUNDLE' "
                    "environment variable should contain the path to the CA "
                    "certificate.  Alternatively, set verify_ssl=False to "
                    "disable certificate verification."
                )
            raise e

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)

    def head(self, url, **kwargs):
        return self.request("HEAD", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)

    def cache_token(self, token, path):
        """Write an OAuth2 token to the cache.

        Parameters
        ----------
        token : OAuth2Token
            Token to be cached.
        path : str
            Path to file containing cached tokens.

        Returns
        -------
        None

        """
        profiles = Session._read_token_cache(path)
        base_url = self._build_url("")

        # Top-level structure if no existing file was found
        if profiles is None:
            profiles = {"profiles": []}

        # Token values to be cached
        token = {
            "accesstoken": token.access_token,
            "refreshtoken": token.refresh_token,
            "tokentype": "bearer",
            "expiry": token.expiration,
        }

        # See if there's an existing profile to update
        matches = [
            (i, p)
            for i, p in enumerate(profiles["profiles"])
            if p["baseurl"] == base_url
        ]
        if matches:
            idx, match = matches[0]
            match["oauthtoken"] = token
            profiles["profiles"][idx] = match
        else:
            profiles["profiles"].append(
                {"baseurl": base_url, "name": None, "oauthtoken": token}
            )

        Session._write_token_cache(profiles, path)

    def read_cached_token(self, path):
        """Read any cached access tokens from disk

        Parameters
        ----------
        path : str
            Path to file containing cached tokens.

        Returns
        -------
        OAuth2Token or None

        """
        # Map from field names in YAML to fields returned by SASLogon
        field_mappings = {
            "accesstoken": "access_token",
            "refreshtoken": "refresh_token",
            "expiry": "expiration",
        }

        profiles = Session._read_token_cache(path)
        url = self._build_url("")

        # Couldn't read any profiles
        if profiles is None:
            return

        # Check each profile for a hostname match and return token if found
        for profile in profiles.get("profiles", []):
            baseurl = profile.get("baseurl", "").lower()

            # Return the cached token
            if baseurl == url.lower():
                data = profile.get("oauthtoken", {})
                token = {
                    field_mappings[k]: v for k, v in data.items() if k in field_mappings
                }
                token = OAuth2Token(**token)

                # Attempt to refresh if cached token has expired
                if token.is_expired:
                    try:
                        token = self._request_token_with_oauth(
                            refresh_token=token.refresh_token
                        )
                    except (
                        exceptions.AuthorizationError,
                        requests.exceptions.HTTPError,
                    ):
                        return

                # If refresh fails, dont return token, allow user to be prompted for login
                if not token.is_expired:
                    return token

    def version_info(self):
        """Get version information from the connected SAS Viya environment

        Returns
        -------
        VersionInfo

        Notes
        -----
        The resulting version information is cached and returned on any subsequent calls.  This
        allows repeatedly checking version information without making redundant network calls to the
        SAS Viya server.

        """
        # The Viya environment isn't changing, so there's no need to repeatedly make
        # service calls to check version information.  If we've already cached the info
        # then just return it.
        if self._version_info:
            return self._version_info

        try:
            # Try to determine if we're talking to Viya 3 or 4
            r = self.get("/licenses/grants")
            release = r.json().get("release")

            # Convert 'V03' and 'V04' to just 3 or 4.
            major_version = int(release.upper().lstrip("V"))

            # No good way to get detailed version info from a Viya 3 environment.
            # At this point, we just assume it's Viya 3.5 and return
            if major_version == 3:
                self._version_info = VersionInfo(major_version)
            else:
                # Endpoint with detailed release info only available for Viya 4
                cadence_info = self.get("/deploymentData/cadenceVersion").json()
                name = cadence_info["cadenceName"]
                release = cadence_info["cadenceVersion"]
                self._version_info = VersionInfo(
                    major_version, cadence=name, release=release
                )
        except HTTPError:
            # Ignore.  We'll return None and (possibly) replace with correct info on subsequent call.
            pass

        return self._version_info

    def _build_url(self, url):
        """Build a complete URL from a path by substituting in session parameters."""
        components = urlsplit(url)

        domain = components.netloc or self._settings["domain"]

        # Add port if a non-standard port was specified
        if self._settings["port"] is not None and ":" not in domain:
            domain = f"{domain}:{self._settings['port']}"

        return urlunsplit(
            [
                components.scheme or self._settings["protocol"],
                domain,
                components.path,
                components.query,
                components.fragment,
            ]
        )

    def _get_authorization_token(
        self,
        token=None,
        username=None,
        password=None,
        client_id=None,
        client_secret=None,
    ):
        """Authenticate with SAS Viya and obtain and access token.

        This method supports multiple authentication methods:

         - an existing OAuth2 token
         - password authentication
         - client credentials
         - Kerberos
         - cached tokens (from previous authorization codes)
         - authorization code

        If authentication using client credentials fails because the
        client_credentials grant type is not allowed the token cache will
        be searched for a valid token.  If no valid token is found, the
        user will be prompted to generate and enter an authorization code.

        Parameters
        ----------
        token : str, optional
            An existing access token to reuse.
        username : str, optional
            Name of the user account to use.  Used for password
            authentication and Kerberos authentication.
        password : str, optional
            Password corresponding to `username`.  Only used for password
            authentication.
        client_id : str, optional
            The id of the client to use during authentication.  Overrides
            the SASCTL_CLIENT_ID environment variable.  Used for password,
            client_credentials, and authorization_code access.
        client_secret : str, optional
            The client secret to use during authentication.  Overrides the
            SASCTL_CLIENT_SECRET environment variable.  Used for password,
            client_credentials, and authorization_code access.

        Returns
        -------
        OAuth2Token

        Raises
        ------
        AuthenticationError
            For invalid username/password combination or invalid client id
            and client secret combination.

        AuthorizationError
            If authorization code is invalid or if refresh token is expired.

        ValueError
            If a client id is specified (either through `client_id` or
            SASCTL_CLIENT ID) but no client secret is provided.

        """
        if token:
            return OAuth2Token(token)

        if username and password:
            return self._request_token_with_oauth(
                username, password, client_id, client_secret
            )

        client_id_provided = (
            client_id is not None or os.getenv("SASCTL_CLIENT_ID") is not None
        )
        client_secret_provided = (
            client_secret is not None or os.getenv("SASCTL_CLIENT_SECRET") is not None
        )

        # Attempt authentication with just client credentials
        if client_id_provided:
            if not client_secret_provided:
                # Authenticating with client credentials should require a client secret.
                raise ValueError(
                    "A client secret must be provided whenever a client id is specified."
                )

            try:
                return self._request_token_with_oauth(
                    client_id=client_id, client_secret=client_secret
                )
            except RuntimeError:
                # If all we have is a client id & client password, there's no way to
                # know whether we're supposed to authenticate directly using the client
                # credentials, or prompt the user for an authorization code.  Try
                # client credential authentication first since it doesn't require user
                # prompts, but if that fails do not raise an exception, we'll try
                # requesting an authorization code.
                logger.debug("Authentication using client credentials is disallowed.")

        # If the necessary Python package is installed try to get a token using Kerberos.
        # NOTE: username may be None since client may already have valid, cached
        #       Kerberos tickets.  If so, Session.username will be updated with the
        #       username from Kerberos.
        if kerberos:
            try:
                return self._request_token_with_kerberos(username)
            except:
                logger.exception(
                    "Encountered an error while attempting Kerberos authorization."
                )
        else:
            logger.debug(
                "Skipping Kerberos authentication - kerberos and winkerberos "
                "packages not found."
            )

        # Before we prompt the user for an authorization code, check if there's already
        # a valid token in the cache from a previous session.
        token = self.read_cached_token(self.PROFILE_PATH)

        if token is not None:
            return token

        # If we got this far, then no password and no kerberos.  Try prompting the user
        # for an authorization code.
        auth_code = self._prompt_for_auth_code(client_id)
        token = self._request_token_with_oauth(
            client_id=client_id, client_secret=client_secret, auth_code=auth_code
        )

        # Cache the token so we don't have to request user input again until the
        # token expires.
        self.cache_token(token, self.PROFILE_PATH)

        return token

    def _prompt_for_auth_code(self, client_id=None):
        """Prompt the user open a URL to generate an auth code.

        Note that this halts program execution until input is received and
        should only be used for interactive sessions.

        Parameters
        ----------
        client_id : str, optional

        Returns
        -------
        str
            Authorization code that can be used to acquire an OAuth2
            access token.

        See Also
        --------
        :meth:`Session.get_oauth_token`

        """
        client_id = client_id or os.environ.get("SASCTL_CLIENT_ID", "sas.ec")

        # User must open this URL in a browser and enter the auth code that's generated.
        url = (
            self._build_url("/SASLogon/oauth/authorize")
            + "?response_type=code&client_id="
            + client_id
        )
        message = (
            f"Please use a web browser to login at the following URL to get your "
            f"authorization code:\n{url}"
        )
        print(message)
        auth_code = input("Authorization Code:")

        return auth_code

    @staticmethod
    def _read_token_cache(path):
        """Read cached OAuth2 access tokens from disk.

        Parameters
        ----------
        path : str
            Path to file containing cached tokens.

        Returns
        -------
        dict or None

        Raises
        ------
        RuntimeError
            If file permissions are too permissive.

        """
        yaml_file = os.path.expanduser(path)

        # See if a token has been cached for the hostname
        if os.path.exists(yaml_file):
            # Get bit flags indicating access permissions
            mode = os.stat(yaml_file).st_mode
            flags = oct(mode)[-3:]

            if flags != "600":
                raise RuntimeError(
                    f"Unable to read profile cache.  The file permissions for "
                    f"{yaml_file} must be configured so that only the file owner has "
                    f"read/write permissions (equivalent to 600 on Linux systems)."
                )

            with open(yaml_file) as f:
                return yaml.safe_load(f)

        return None

    def _request_token_with_consul(self, consul_token, client_id=None):
        """Request an OAuth token from Consul.

        This functionality is reserved for system administrators as access
        to the Consul token is restricted.

        Parameters
        ----------
        consul_token : str
            A Consul ACL token
        client_id : str, optional
            The name of the client being used to access the service.

        Returns
        -------
        OAuth2Token

        Raises
        ------
        AuthenticationError
            If unable to retrieve an access token for any reason.

        """
        client_id = client_id or ""

        headers = {"X-Consul-Token": consul_token}
        params = {"callback": False, "serviceId": client_id}

        url = self._build_url("/SASLogon/oauth/clients/consul")
        response = super(Session, self).post(
            url,
            headers=headers,
            params=params,
            verify=self.verify,
        )

        try:
            response.raise_for_status()
            data = response.json()
            return OAuth2Token(**data)
        except requests.HTTPError as e:
            raise exceptions.AuthenticationError(
                msg="Failed to authenticate using Consul token."
            ) from e

    def _request_token_with_kerberos(self, username=None):
        """Authenticate with a Kerberos ticket.

        Parameters
        ----------
        username : str, optional

        Returns
        -------
        OAuth2Token

        Raises
        ------
        RuntimeError
            If required kerberos package is not installed.

        ValueError
            If there is any issue with the server's response to the
            authorization request.

        """
        if kerberos is None:
            raise RuntimeError(
                "Kerberos package not found.  Run 'pip "
                "install sasctl[kerberos]' to install."
            )

        flags = kerberos.GSS_C_MUTUAL_FLAG | kerberos.GSS_C_SEQUENCE_FLAG
        service = "HTTP@%s" % self._settings["domain"]

        logger.info("Attempting Kerberos authentication to %s", service)

        url = self._build_url(
            "/SASLogon/oauth/authorize?client_id=sas.tkmtrb&response_type=token"
        )

        # Get Kerberos challenge
        r = self.get(url, allow_redirects=False, verify=self.verify)

        if r.status_code != 401:
            raise ValueError(
                "Kerberos challenge response not received.  "
                "Expected HTTP 401 but received %s" % r.status_code
            )

        if "www-authenticate" not in r.headers:
            raise ValueError(
                "Kerberos challenge response not received.  "
                "'WWW-Authenticate' header not received."
            )

        if "Negotiate" not in r.headers["www-authenticate"]:
            raise ValueError(
                "Kerberos challenge response not received.  "
                "'WWW-Authenticate' header contained '%s', "
                "expected 'Negotiate'." % r.headers["www-authenticate"]
            )

        # Initialize a request to KDC for a ticket to access the service.
        _, context = kerberos.authGSSClientInit(
            service, principal=username, gssflags=flags
        )

        # Send the request.
        # NOTE: empty-string parameter required for initial call.
        kerberos.authGSSClientStep(context, "")

        # Get the KDC response
        auth_header = "Negotiate %s" % kerberos.authGSSClientResponse(context)

        # Get the user that was used for authentication
        username = kerberos.authGSSClientUserName(context)

        # Drop @REALM from username and store
        if username is not None:
            self._settings["username"] = username.rsplit("@", maxsplit=1)[0]

        # Response to Kerberos challenge with ticket
        r = self.get(
            url,
            headers={"Authorization": auth_header},
            allow_redirects=False,
            verify=self.verify,
        )

        if "Location" not in r.headers:
            raise ValueError(
                "Invalid authentication response." "'Location' header not received."
            )

        match = re.search("(?<=access_token=)[^&]*", r.headers["Location"])

        if match is None:
            raise ValueError(
                "Invalid authentication response.  'Location' "
                "header does not contain an access token."
            )

        # Extract access token and return as an Oauth token
        return OAuth2Token(match.group(0))

    def _request_token_with_oauth(
        self,
        username=None,
        password=None,
        client_id=None,
        client_secret=None,
        auth_code=None,
        refresh_token=None,
    ):
        """Request a token from the SAS SASLogon service.

        Supports four different flows:
         - authenticate with a username & password and receive a token
         - authenticate with a client id & secret and receive a token
         - provide an authorization code and receive a token
         - provide a refresh token and receive a new token

        Parameters
        ----------
        username : str, optional
            Username of the user in SAS Viya.  Required for password
            authentication flow.
        password : str, optional
            Password of the user in SAS Viya.  Required for password
            authentication flow.
        client_id : str, optional
            Client ID of a client registered with SAS Viya.  Required for
            client credentials flow, but will use
            a default client ID if not provided during password or
            authorization code flows.
        client_secret : str, optional
            Client secret of a client registered with SAS Viya.  Required
            for client credentials flow, but will use a default if not
            provided during password or authorization code flows.
        auth_code : str, optional
            An authorization code obtained by the user after
            authenticating with SAS Viya.  Required for authorization
            code flow.
        refresh_token : str, optional
            A refresh token obtained during a previous authorization
            request.  Required if requesting a refreshed access token.

        Returns
        -------
        OAuth2Token

        Raises
        ------
        ValueError
            If no valid parameter combination is provided.

        RuntimeError
            If the requested grant type is not allowed for the given
            `client_id`.

        AuthenticationError
            For invalid username/password combination or invalid
            client id/secret combination.

        AuthorizationError
            If authorization code is invalid or if refresh token is expired.

        requests.HTTPError
            For any unexpected/unhandled HTTP error code (e.g. HTTP 404)

        requests.ConnectionError
            If there's an issue establishing a connection with the server.

        """
        client_id = client_id or os.getenv("SASCTL_CLIENT_ID", "sas.ec")
        client_secret = client_secret or os.getenv("SASCTL_CLIENT_SECRET", "")

        # Order in which parameters are checked is important.  Passing client
        # credentials could indicate password, auth code, or client credential
        # authentication.  Only use client credential flow if first two are
        # ruled out.
        if username:
            anchor = "#password"
            data = {
                "grant_type": "password",
                "username": username,
                "password": password,
            }
            logger.debug("Attempting password authentication as user '%s'.", username)
        elif auth_code:
            anchor = "#authorization_code"
            data = {"grant_type": "authorization_code", "code": auth_code}
            logging.debug("Attempting authorization using an auth code.")
        elif client_secret:
            anchor = "#client_credentials"
            data = {"grant_type": "client_credentials"}
            logger.debug(
                "Attempting client credential authentication as client '%s'.", client_id
            )
        elif refresh_token:
            anchor = ""
            data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
        else:
            raise ValueError("At least one set of parameters must be provided.")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        url = self._build_url(f"/SASLogon/oauth/token{anchor}")
        response = super(Session, self).post(
            url,
            headers=headers,
            data=data,
            auth=(client_id, client_secret),
            verify=self.verify,
        )

        try:
            data = response.json()
        except json.JSONDecodeError:
            # A valid response (and some HTTP errors) should contain valid JSON.  In
            # the extremely unlikely event that an HTTP error is returned that doesn't
            # include JSON (e.g. HTTP 404) just raise the HTTP error.
            response.raise_for_status()

        # If request failed for a known reason, raise a user-friendly error message.
        if (
            response.status_code == 400
            and auth_code is not None
            and "Invalid authorization code" in data.get("error_description", "")
        ):
            raise exceptions.AuthorizationError(
                f"Invalid authorization code: {auth_code}."
            )

        if response.status_code == 401:
            # Response is the same if either username/password or client id/secret is
            # invalid ('{"error":"unauthorized","error_description":"Bad credentials"}')
            # We're assuming that if a username was provided, that's probably the problem.
            # NOTE: this does mean that valid credentials with incorrectly set client
            #       id & secret results in an error message that blames the username
            #       and password combination.
            if "bad credentials" in data.get("error_description", "").lower():
                if username is not None:
                    raise exceptions.AuthenticationError(username)

                raise exceptions.AuthenticationError(msg="Invalid client id or secret.")

            # If client used is not allowed to authenticate using the requested method
            # an error is returned.  Example:
            # '{"error":"invalid_client","error_description":"Unauthorized grant type: password"}'
            if "unauthorized grant type" in data.get("error_description", "").lower():
                raise RuntimeError(data["error_description"])

            if refresh_token is not None and data.get("error", "") == "invalid_token":
                raise exceptions.AuthorizationError(
                    "Refresh token is incorrect, expired, or revoked."
                )

        # Raise a generic exception for any other issues
        response.raise_for_status()

        return OAuth2Token(**data)

    @staticmethod
    def _write_token_cache(profiles, path):
        """

        Parameters
        ----------
        profiles : dict
        path : str

        Returns
        -------
        None

        """
        yaml_file = os.path.expanduser(path)

        # Create parent .sas folder if needed
        sas_dir = os.path.dirname(yaml_file)
        if not os.path.exists(sas_dir):
            os.mkdir(sas_dir)

        with open(yaml_file, "w") as f:
            yaml.dump(profiles, f)

        # Get bit flags indicating access permissions
        mode = os.stat(yaml_file).st_mode
        flags = oct(mode)[-3:]

        # Ensure access to file is restricted
        if flags != "600":
            os.chmod(yaml_file, 0o600)

    def __enter__(self):
        super(Session, self).__enter__()

        # Make this the current session
        # self._old_session = current_session()
        # current_session(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous current session
        current_session(self._old_session)

        super(Session, self).__exit__()

    def __str__(self):
        return (
            "{class_}(hostname='{hostname}', username='{username}', "
            "protocol='{protocol}', verify_ssl={verify})".format(
                class_=type(self).__name__,
                hostname=self.hostname,
                username=self.username,
                protocol=self._settings.get("protocol"),
                verify=self.verify,
            )
        )


class PageIterator:
    """Iterates through a collection that must be "paged" from the server.

    Pages contain a batch of items from the overall collection.  Iterates the
    series of pages and returns a single batch of items each time `next()` is
    called.

    Parameters
    ----------
    obj : RestObj
        An instance of `RestObj` containing any initial items and a link to
        retrieve additional items.
    session : Session
        The `Session` instance to use for requesting additional items.  Defaults
        to current_session()
    threads : int
        Number of threads allocated to downloading additional pages.

    Yields
    ------
    List[RestObj]
        Items contained in the current page

    """

    def __init__(self, obj, session=None, threads=4):
        self._num_threads = threads

        # Session to use when requesting items
        self._session = session or current_session()
        self._pool = None
        self._requested = []

        link = get_link(obj, "next")

        # Dissect the "next" link so it can be reformatted and used by
        # parallel threads
        if link is None:
            self._min_queue_len = 0
            self._start = 0
            self._limit = 0
        else:
            link = link["href"]
            start = re.search(r"(?<=start=)[\d]+", link)
            limit = re.search(r"(?<=limit=)[\d]+", link)

            # Construct a new link with format params
            # Result is "/spam/spam?start={start}&limit={limit}"
            link = (
                link[: start.start()]
                + "{start}"
                + link[start.end() : limit.start()]
                + "{limit}"
                + link[limit.end() :]
            )

            self._start = int(start.group())
            self._limit = int(limit.group())

            # Length at which to beging requesting new items from the server
            self._min_queue_len = self._limit

        self._next_link = link

        # Store the current items to iterate over
        self._obj = obj

    def __next__(self):
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num_threads
            )

        # Send request for new pages if we don't have enough cached
        num_req_needed = self._num_threads - len(self._requested)
        for _ in range(num_req_needed):
            self._requested.append(self._pool.submit(self._request_async, self._start))
            self._start += self._limit

        # If this is the first time next() has been called, return the items
        # contained in the initial page.
        if self._obj is not None:
            result = [RestObj(x) for x in self._obj["items"]]
            self._obj = None
            return result

        # Make sure the next page has been received
        if self._requested:
            items = self._requested.pop(0).result()

            if not items:
                raise StopIteration

            return items

        raise StopIteration

    def __iter__(self):
        # All Iterators are also Iterables
        return self

    def _request_async(self, start):
        """Used by worker threads to retrieve next batch of items."""

        if self._next_link is None:
            return []

        # Format the link to retrieve desired batch
        link = self._next_link.format(start=start, limit=self._limit)
        r = get(link, format="json", session=self._session)
        r = RestObj(r)
        return [RestObj(x) for x in r["items"]]


class PagedItemIterator:
    """Iterates through a collection that must be "paged" from the server.

    Uses `PageIterator` to transparently download pages of items from the server
    as needed.

    Parameters
    ----------
    obj : RestObj
        An instance of `RestObj` containing any initial items and a link to
        retrieve additional items.
    session : Session
        The `Session` instance to use for requesting additional items.  Defaults
        to current_session()
    threads : int
        Number of threads allocated to downloading additional items.

    Yields
    ------
    RestObj

    Notes
    -----
    Value returned by len() is an approximate count of the items available.  The actual
    number of items returned may be greater than or less than this number.

    See Also
    --------
    PageIterator

    """

    def __init__(self, obj, session=None, threads=4):
        # Iterates over whole pages of items
        self._pager = PageIterator(obj, session, threads)

        # Store items from latest page that haven't been returned yet.
        self._cache = []

        # Total number of items to iterate over
        if "count" in obj:
            # NOTE: "count" may be an (over) estimate of the number of items available
            #       since some may be inaccessible due to user permissions & won't
            #       actually be returned.
            self._count = int(obj.count)
        else:
            self._count = len(obj["items"])

    def __len__(self):
        return self._count

    def __next__(self):
        # Get next page of items if we're currently out
        if not self._cache:
            self._cache = next(self._pager)

        # Return the next item
        if self._cache:
            self._count -= 1
            return self._cache.pop(0)

        raise StopIteration()

    def __iter__(self):
        return self


class PagedListIterator:
    """Iterates over an instance of PagedList

    Parameters
    ----------
    l : list-like

    """

    def __init__(self, l):
        self.__list = l
        self.__index = 0

    def __next__(self):
        if self.__index >= len(self.__list):
            raise StopIteration

        try:
            item = self.__list[self.__index]
            self.__index += 1
            return item
        except IndexError:
            # Because PagedList length is approximate, iterating can result in
            # indexing outside the array.  Just stop the iteration if that occurs.
            raise StopIteration

    def __iter__(self):
        return self


class PagedList(list):
    """List that dynamically loads items from the server.

    Parameters
    ----------
    obj : RestObj
        An instance of `RestObj` containing any initial items and a link to
        retrieve additional items.
    session : Session, optional
        The `Session` instance to use for requesting additional items.  Defaults
        to current_session()
    threads : int, optional
        Number of threads allocated to loading additional items.

    Notes
    -----
    Value returned by len() is an approximate count of the items available.  The actual
    length is not known until all items have been pulled from the server.

    See Also
    --------
    PagedItemIterator

    """

    def __init__(self, obj, session=None, threads=4):
        super(PagedList, self).__init__()
        self._paged_items = PagedItemIterator(obj, session=session, threads=threads)

        # Go ahead and add the items that were initially returned.
        # Do this by "paging" so iterator remains at the correct spot.
        for _ in range(len(obj["items"])):
            self.append(next(self._paged_items))

        # Assume that server has more items available
        self._has_more = True

    def __len__(self):
        if self._has_more:
            # Estimate the total length as items downloaded + items still on server
            return super(PagedList, self).__len__() + len(self._paged_items)
        else:
            # We've pulled everything from the server, so we have an exact length now.
            return super(PagedList, self).__len__()

    def __iter__(self):
        return PagedListIterator(self)

    def __getslice__(self, i, j):
        # Removed from Py3.x but still implemented in CPython built-in list
        # Override to ensure __getitem__ is used instead.
        return self.__getitem__(slice(i, j))

    def __getitem__(self, item):
        if hasattr(item, "stop"):
            # `item` is a slice
            # if no stop was specified, assume full length
            idx = item.stop or len(self)
        else:
            idx = int(item)

            # Support negative indexing.  Need to load items up to len() - idx.
            if idx < 0:
                idx = len(self) + idx

        try:
            # Iterate through server-side pages until we've loaded
            # the item at the requested index.
            while super(PagedList, self).__len__() <= idx:
                n = next(self._paged_items)
                self.append(n)

        except StopIteration:
            # We've hit the end of the paging so the server has no more items to retrieve.
            self._has_more = False

        # Get the item from the list
        return super(PagedList, self).__getitem__(item)

    def __repr__(self):
        string = super(PagedList, self).__repr__()

        # If the list has more "items" than are stored in the underlying list
        # then there are more downloads to make.
        if len(self) - super(PagedList, self).__len__() > 0:
            string = string.rstrip("]") + ", ...]"

        return string


class VersionInfo:
    """Stores the version information for a SAS Viya environment.

    Parameters
    ----------
    major : int
        Major version number (e.g. 3 for Viya 3.5 or 4 for Viya 4.0)
    minor : int
        Minor version number (e.g. 5 for Viya 3.5 or 0 for Viya 4.0)
    cadence : str, optional
        Release cadence for Viya 4.  Should be one of 'stable' or 'LTS'.
    release : str, optional
        Release number for Viya 4.  Two formats are currently possible:
         - YYYY.R.U where R is the LTS release number in YYYY and U is the updates since R
         - YYYY.MM where MM is the month of the release.

    """

    def __init__(self, major, minor=None, cadence=None, release=None):
        self._major = major
        self._minor = minor if minor else 5 if major == 3 else 0
        self._cadence = str(cadence) if cadence else None
        self._release = str(release) if release else None

    def __ge__(self, other):
        return self > other or self == other

    def __gt__(self, other):
        return self._compare(other) > 0

    def __lt__(self, other):
        return self._compare(other) < 0

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        return self._compare(other) == 0

    def __float__(self):
        return float(self._major) + 0.1 * (self._minor or 0)

    def __hash__(self):
        return hash((self.major, self.minor, self.release))

    def _compare(self, other):
        """Compare and return -1/0/1 indicating lt/eq/gt.

        Parameters
        ----------
        other : any

        Returns
        -------
        int
            -1 if < `other`, 0 if equal, and 1 if > `other`.

        """
        # Compare Major/Minor versions (e.g. Viya 3.5 < Viya 4.0)
        if float(self) > float(other):
            return 1
        if float(self) < float(other):
            return -1

        # If comparing two Viya 4 versions, may need to check actual release number  to determine order
        if self.release and getattr(other, "release", None):
            if self._release == other.release:
                return 0

            parts = self.release.split(".")
            other_parts = other.release.split(".")

            # Release format was changed from YYYY.r.u to YYYY.MM so any release with 2 '.' is older than
            # a release with 1 '.'
            if len(parts) == 2 and len(other_parts) == 3:
                return 1
            if len(parts) == 3 and len(other_parts) == 2:
                return -1

            # If we got this far then both version numbers should have the same release format.
            # Earlier release is the one with at least one component part that is lower
            for a, b in zip(parts, other_parts):
                if int(a) < int(b):
                    return -1
                if int(a) > int(b):
                    return 1

        # Either other doesn't have a .release or we compared all components of .release and
        # they were all equal
        return 0

    def __repr__(self):
        name = f"{self.__class__.__name__}(major={self._major}, minor={self._minor}"

        if self._cadence:
            name += ", cadence='%s'" % self._cadence

        if self._release:
            name += ", release='%s'" % self._release

        return name + ")"

    @property
    def cadence(self):
        return self._cadence

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def release(self):
        return self._release


def is_uuid(id_):
    try:
        UUID(str(id_))
        return True
    except (ValueError, TypeError):
        return False


def get(path, **kwargs):
    """Send a GET request.

    Parameters
    ----------
    path : str
        The path portion of the URL.
    kwargs : any
        Passed to `request`.

    Returns
    -------
    RestObj or None
        The results or None if the resource was not found.

    """
    try:
        return request("get", path, **kwargs)
    except HTTPError as e:
        if e.code == 404:
            return None  # Resource not found
        raise e


def head(path, **kwargs):
    """Send a HEAD request.

    Parameters
    ----------
    path : str
        The path portion of the URL.
    kwargs : any
        Passed to `request`.

    Returns
    -------
    RestObj

    """
    return request("head", path, **kwargs)


def post(path, **kwargs):
    """Send a POST request.

    Parameters
    ----------
    path : str
        The path portion of the URL.
    kwargs : any
        Passed to `request`.

    Returns
    -------
    RestObj

    """
    return request("post", path, **kwargs)


def put(path, item=None, **kwargs):
    """Send a PUT request.

    Parameters
    ----------
    path : str
        The path portion of the URL.
    item : RestObj, optional
        A existing object to PUT.  If provided, ETag and Content-Type headers
        will automatically be specified.
    kwargs : any
        Passed to `request`.

    Returns
    -------
    RestObj

    """
    # If call is in the format put(url, RestObj), automatically fill in header
    # information
    if item is not None and isinstance(item, RestObj):
        get_headers = getattr(item, "_headers", None)
        if get_headers is not None:
            # Update the headers param if it was specified
            headers = kwargs.pop("headers", {})
            headers.setdefault("If-Match", get_headers.get("etag"))
            headers.setdefault("Content-Type", get_headers.get("content-type"))
            return request("put", path, json=item, headers=headers)

    return request("put", path, **kwargs)


def delete(path, **kwargs):
    """Send a DELETE request.

    Parameters
    ----------
    path : str
        The path portion of the URL.
    kwargs : any
        Passed to `request`.

    Returns
    -------
    RestObj

    """
    return request("delete", path, **kwargs)


def request(verb, path, session=None, format="auto", **kwargs):
    """Send an HTTP request with a session.

    Parameters
    ----------
    verb : str
        A valid HTTP request verb.
    path : str
        Path portion of URL to request.
    session : Session, optional
        Defaults to `current_session()`.
    format : {'auto', 'rest', 'response', 'content', 'json', 'text'}
        The format of the return response.  Defaults to `auto`.
        rest: `RestObj` constructed from JSON.
        response: the raw `Response` object.
        content: Response.content
        json: Response.json()
        text: Response.text
        auto: `RestObj` constructed from JSON if possible, otherwise same as
              `text`.
    kwargs : any
        Additional arguments are passed to the session `request` method.

    Returns
    -------

    """
    session = session or current_session()

    if session is None:
        raise TypeError("No `Session` instance found.")

    format = "auto" if format is None else str(format).lower()
    if format not in ("auto", "response", "content", "text", "json", "rest"):
        raise ValueError

    response = session.request(verb, path, **kwargs)

    if 400 <= response.status_code <= 599:
        raise HTTPError(
            response.url, response.status_code, response.text, response.headers, None
        )

    # Return the raw response if requested
    if format == "response":
        return response
    if format == "json":
        return response.json()
    if format == "text":
        return response.text
    if format == "content":
        return response.content
    try:
        obj = _unwrap(response.json())

        # ETag is required to update any object
        # May not be returned on all responses (e.g. listing
        # multiple objects)
        if isinstance(obj, RestObj):
            obj._headers = response.headers
        return obj
    except ValueError:
        if format == "rest":
            return RestObj()
        return response.text


def get_link(obj, rel):
    """Get link information from a resource.

    Parameters
    ----------
    obj : dict
    rel : str

    Returns
    -------
    dict

    """
    if isinstance(obj, dict) and "links" in obj:
        if isinstance(obj["links"], dict):
            return obj["links"].get(rel)

        links = [l for l in obj.get("links", []) if l.get("rel") == rel]
        if not links:
            return None
        if len(links) == 1:
            return links[0]
        return links

    if isinstance(obj, dict) and "rel" in obj and obj["rel"] == rel:
        # Object is already a link, just return it
        return obj


def request_link(obj, rel, **kwargs):
    """Request a link from a resource.

    Parameters
    ----------
    obj : dict
    rel : str
    kwargs : any
        Passed to :function:`request`

    Returns
    -------

    """
    link = get_link(obj, rel)

    if link is None:
        raise ValueError("Link '%s' not found in object %s." % (rel, obj))

    return request(link["method"], link["href"], **kwargs)


def uri_as_str(obj):
    """Get the URI of a resource in string format.

    Parameters
    ----------
    obj : str or dict
        Strings are assumed to be URIs and are returned as is.  Dictionary
        objects will be checked for a `self` URI and returned if found.

    Returns
    -------
    str
        Resource URI or None if not found

    """
    if isinstance(obj, dict):
        link = get_link(obj, "self")
        if isinstance(link, dict):
            return link.get("uri")

    return obj


def _unwrap(json):
    """Converts a JSON response to one or more `RestObj` instances.

    If the JSON contains a .items property, only those items are converted and returned.

    Parameters
    ----------
    json

    Returns
    -------

    """
    if "items" in json:
        if len(json["items"]) == 1:
            return RestObj(json["items"][0])
        if len(json["items"]) > 1:
            return PagedList(RestObj(json))
        return []

    return RestObj(json)


def _build_crud_funcs(path, single_term=None, plural_term=None, service_name=None):
    """Utility method for defining simple functions to perform CRUD operations on a REST endpoint.

    Parameters
    ----------
    path : str
        URL path to use for the requests
    single_term : str
        English name of the item being manipulated. Defaults to `plural_term`.
    plural_term : str
        English name of the items being manipulated. Defaults to the last segment of `path`.
    service_name : str
        Name of the service under which the command will be listed in the `sasctl` CLI.  Defaults to `plural_term`.

    Returns
    -------
    functions : tuple
        tuple of CRUD functions: list_items, get_item, update_item, delete_item


    Examples
    --------

    >>> list_spam, get_spam, update_spam, delete_spam = _build_crud_funcs('/spam')

    """

    @sasctl_command("list")
    def list_items(filter=None):
        """List all {items} available in the environment.

        Parameters
        ----------
        filter : str, optional

        Returns
        -------
        list
            A list of dictionaries containing the {items}.

        Notes
        -----
        See the filtering_ reference for details on the `filter` parameter.

        .. _filtering: https://developer.sas.com/reference/filtering/

        """
        params = "filter={}".format(filter) if filter is not None else {}

        results = get(path, params=params)
        return results if isinstance(results, list) else [results]

    @sasctl_command("get")
    def get_item(item, refresh=False):
        """Returns a {item} instance.

        Parameters
        ----------
        item : str or dict
            Name, ID, or dictionary representation of the {item}.
        refresh : bool, optional
            Obtain an updated copy of the {item}.

        Returns
        -------
        RestObj or None
            A dictionary containing the {item} attributes or None.

        Notes
        -------
        If `item` is a complete representation of the {item} it will be returned unless `refresh` is set.  This
        prevents unnecessary REST calls when data is already available on the client.

        """

        # If the input already appears to be the requested object just return it, unless
        # a refresh of the data was explicitly requested.
        if isinstance(item, dict) and all(k in item for k in ("id", "name")):
            if refresh:
                item = item["id"]
            else:
                return item

        if is_uuid(item):
            return get(path + "/{id}".format(id=item))
        results = list_items(filter='eq(name, "{}")'.format(item))

        # Not sure why, but as of 19w04 the filter doesn't seem to work.
        for result in results:
            if result["name"] == str(item):
                # Make a request for the specific object so that ETag is
                # included, allowing updates.
                if get_link(result, "self"):
                    return request_link(result, "self")

                id_ = result.get("id", result["name"])
                return get(path + "/{id}".format(id=id_))

        return None

    @sasctl_command("update")
    def update_item(item):
        """Updates a {item} instance.

        Parameters
        ----------
        item : dict

        Returns
        -------
        None

        """

        headers = getattr(item, "_headers", None)
        if headers is None or headers.get("etag") is None:
            raise ValueError("Could not find ETag for update of %s." % item)

        id_ = getattr(item, "id", None)
        if id_ is None:
            raise ValueError("Could not find property `id` for update of %s." % item)

        headers = {
            "If-Match": item._headers.get("etag"),
            "Content-Type": item._headers.get("content-type"),
        }

        return put(path + "/%s" % id_, json=item, headers=headers)

    @sasctl_command("delete")
    def delete_item(item):
        """Deletes a {item} instance.

        Parameters
        ----------
        item

        Returns
        -------
        None

        """

        # Try to find the item if the id can't be found
        if not (isinstance(item, dict) and "id" in item):
            item = get_item(item)

        if isinstance(item, dict) and "id" in item:
            item = item["id"]

        if is_uuid(item):
            return delete(path + "/{id}".format(id=item))
        raise ValueError("Unrecognized id '%s'" % item)

    # Pull object name from path if unspecified (many paths end in /folders or /repositories).
    plural_term = plural_term or str(path).split("/")[-1]
    single_term = single_term or plural_term
    service_name = service_name or plural_term
    service_name = service_name.replace(" ", "_")

    for func in [list_items, get_item, update_item, delete_item]:
        func.__doc__ = func.__doc__.format(item=single_term, items=plural_term)
        func._cli_service = service_name

        prefix = func.__name__.split("_")[0] + "_"
        suffix = plural_term if prefix == "list_" else single_term
        func.__name__ = prefix + suffix

    return list_items, get_item, update_item, delete_item


def _build_is_available_func(service_root):
    def is_available():
        """Checks if the service is currently available.

        Returns
        -------
        bool

        """
        response = current_session().head(service_root + "/")
        return response.status_code == 200

    return is_available


@versionadded(version="1.5.6")
def platform_version():
    """Get the version of the SAS Viya platform to which sasctl is connected.

    Returns
    -------
    string : {'3.5', '4.0'}

        SAS Viya version number

    """
    warnings.warn(
        "platform_version() has been deprecated and will be removed in a future version.  "
        "Please use Session.version_info() instead.",
        DeprecationWarning,
    )

    from .services import model_repository as mr

    response = mr.info()
    buildVersion = response.get("build")["buildVersion"]
    try:
        if buildVersion[0:4] == "3.7.":
            return "3.5"
        elif float(buildVersion[0:4]) >= 3.10:
            return "4.0"
    except ValueError:
        pass  # Version could not be found.  Return None instead.
