#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import json
import netrc
import os
import re
import ssl
import sys
import warnings
from uuid import UUID, uuid4

import requests, requests.exceptions
from requests.adapters import HTTPAdapter, DEFAULT_POOLBLOCK
from six.moves.urllib.parse import urlsplit, urlunsplit
from six.moves.urllib.error import HTTPError

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
from . import exceptions

logger = logging.getLogger(__name__)

_session = None


def pformat(text):
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
        string = string.decode('utf-8') if is_bytes else string
        string = re.sub(pattern, repl, string)
        string = string.encode('utf-8') if is_bytes else string
    except UnicodeDecodeError:
        pass
    return string


def _filter_password(r):
    if hasattr(r, 'body') and r.body is not None:
        # Filter password from 'grant_type=password&username=<user>&password=<password>' during Post to Logon service.
        r.body = _redact(r"(?<=&password=)([^&]*)\b", '*****', r.body)

        # Filter client secret {"client_secret": "<password>"}
        r.body = _redact('(?<=client_secret": ")[^"]*', '*****', r.body)
    return r


def _filter_token(r):
    # Redact "Bearer <token>" in Authorization headers
    if hasattr(r, 'headers') and 'Authorization' in r.headers:
        r.headers['Authorization'] = _redact(r'(?<=Bearer ).*', '[redacted]',
                                             r.headers['Authorization'])

    # Redact "Basic <base64 encoded pw> in Authorization headers.  This covers client ids & client secrets.
    if hasattr(r, 'headers') and 'Authorization' in r.headers:
        r.headers['Authorization'] = _redact(r'(?<=Basic ).*', '[redacted]',
                                             r.headers['Authorization'])

    # Redact Consul token from headers.  Should only appear when registering a new client
    if hasattr(r, 'headers') and 'X-Consul-Token' in r.headers:
        r.headers['X-Consul-Token'] = '[redacted]'

    # Redact "access_token":"<token>" in response from Logon service
    if hasattr(r, '_content'):
        r._content = _redact('(?<=access_token":")[^"]*', '[redacted]',
                             r._content)

    return r


DEFAULT_FILTERS = [
    _filter_password,
    _filter_token
]


def current_session(*args, **kwargs):
    global _session

    # Explicitly set or clear the current session
    if len(args) == 1 and (isinstance(args[0], Session) or args[0] is None):
        _session = args[0]
    # Create a new session
    elif len(args):
        _session = Session(*args, **kwargs)

    return _session


class HTTPBearerAuth(requests.auth.AuthBase):
    # Taken from https://github.com/kennethreitz/requests/issues/4437

    def __init__(self, token):
        self.token = token

    def __eq__(self, other):
        return self.token == getattr(other, 'token', None)

    def __ne__(self, other):
        return not self == other

    def __call__(self, r):
        r.headers['Authorization'] = 'Bearer ' + self.token
        return r


class RestObj(dict):
    def __getattr__(self, item):
        # Only called when __getattribute__ failed to find the attribute
        # Return the item from underlying dictionary if possible.
        if item in self:
            result = self[item]

            if isinstance(result, dict):
                return RestObj(result)
            else:
                return result
        else:
            raise AttributeError("'%s' object has no attribute '%s'" % (
                self.__class__.__name__, item))

    def __repr__(self):
        headers = getattr(self, '_headers', {})

        return "%s(headers=%r, data=%s)" % (
            self.__class__, headers, super(RestObj, self).__repr__())

    def __str__(self):
        if 'name' in self:
            return str(self['name'])
        if 'id' in self:
            return str(self['id'])
        return repr(self)


class SSLContextAdapter(HTTPAdapter):
    """HTTPAdapter that uses the default SSL context on the machine."""

    def __init__(self, *args, **kwargs):
        self.assert_hostname = kwargs.pop('assert_hostname', True)
        requests.adapters.HTTPAdapter.__init__(self, *args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.check_hostname = self.assert_hostname
        kwargs['ssl_context'] = context
        kwargs['assert_hostname'] = self.assert_hostname
        return super(SSLContextAdapter, self).init_poolmanager(*args, **kwargs)


class Session(requests.Session):
    """Establish a connection to a SAS Viya server.

    Parameters
    ----------
    hostname : str or swat.CAS
        Name of the server to connect to or an established swat.CAS session.
    username : str, optional
        Username for authentication.  Not required if `host` is a CAS
        connection or if Kerberos is used.  If using Kerberos and an explicit
        username is desired, maybe be a string in 'user@REALM' format.
    password : str, optional
        Password for authentication.  Not required when `host` is a CAS
        connection, `authinfo` is provided, or Kerberos is used.
    authinfo : str, optional
        Path to a .authinfo or .netrc file from which credentials should be
        pulled.
    protocol : {'http', 'https'}
        Whether to use HTTP or HTTPS connections.  Defaults to `https`.
    port : int, optional
        Port number for the connection if a non-standard port is used.
        Defaults to 80 or 443 depending on `protocol`.
    verify_ssl : bool
        Whether server-side SSL certificates should be verified.  Defaults
        to true.  Ignored for HTTP connections.

    Attributes
    ----------
    message_log
    filters

    """
    def __init__(self, hostname,
                 username=None,
                 password=None,
                 authinfo=None,
                 protocol=None,
                 port=None,
                 verify_ssl=None):
        super(Session, self).__init__()

        # Determine whether or not server SSL certificates should be verified.
        if verify_ssl is None:
            verify_ssl = os.environ.get('SSLREQCERT', 'yes')
            verify_ssl = str(verify_ssl).lower() not in ('no', 'false')

        self._id = uuid4().hex
        self.message_log = logger.getChild('session.%s' % self._id)

        # If certificate path has already been set for SWAT package, make
        # Requests module reuse it.
        for k in ['SSLCALISTLOC', 'CAS_CLIENT_SSL_CA_LIST']:
            if k in os.environ:
                os.environ['REQUESTS_CA_BUNDLE'] = os.environ[k]
                break

        # If certificate path hasn't been specified in either environment
        # variable, replace the default adapter with one that will use the
        # machine's default SSL _settings.
        if 'REQUESTS_CA_BUNDLE' not in os.environ:
            if verify_ssl:
                # Skip hostname verification if IP address specified instead
                # of DNS name.  Prevents error from urllib3
                from urllib3.util.ssl_ import is_ipaddress
                verify_hostname = not is_ipaddress(hostname)
                adapter = SSLContextAdapter(assert_hostname=verify_hostname)

                self.mount('https://', adapter)

            else:
                # Every request will generate an InsecureRequestWarning
                from urllib3.exceptions import InsecureRequestWarning
                warnings.simplefilter('default', InsecureRequestWarning)

        self.filters = DEFAULT_FILTERS

        # Used for context manager
        self._old_session = None

        # Reuse an existing CAS connection if possible
        if swat and isinstance(hostname, swat.CAS):
            if isinstance(hostname._sw_connection,
                          swat.cas.rest.connection.REST_CASConnection):
                import base64

                # Use the httpAddress action to retieve information about REST endpoints
                httpAddress = hostname.get_action('builtins.httpAddress')
                address = httpAddress()
                domain = address.virtualHost
                # httpAddress action may return virtualHost = ''
                # if this happens, try the CAS host
                if not domain:
                    domain = hostname._sw_connection._current_hostname
                protocol = address.protocol
                port = address.port

                # protocol = hostname._protocol
                auth = hostname._sw_connection._auth.decode('utf-8').replace(
                    'Basic ', '')
                username, password = base64.b64decode(auth).decode('utf-8').split(
                    ':')
                # domain = hostname._hostname
            else:
                raise ValueError(
                    "A 'swat.CAS' session can only be reused when it's connected via the REST APIs.")
        else:
            domain = str(hostname)

        self._settings = {'protocol': protocol or 'https',
                         'domain': domain,
                         'port': port,
                         'username': username,
                         'password': password
                          }

        if self._settings['password'] is None:
            # Try to get credentials from .authinfo or .netrc files.
            # If no file path was specified, the default locations will
            # be checked.
            if 'swat' in sys.modules:
                auth = swat.utils.authinfo.query_authinfo(domain, user=username,
                                                          path=authinfo)
                self._settings['username'], self._settings[
                    'password'] = auth.get('username'), auth.get('password')

            # Not able to load credentials using SWAT.  Try Netrc.
            # TODO: IF a username was specified, verify that the credentials found
            #       are for that username.
            if self._settings['password'] is None:
                try:
                    parser = netrc.netrc(authinfo)
                    values = parser.authenticators(domain)
                    if values:
                        self._settings['username'], \
                        _, \
                        self._settings['password'] = values
                except (OSError, IOError):
                    pass  # netrc throws if $HOME is not set

        self.verify = verify_ssl
        self.auth = HTTPBearerAuth(self.get_token())

        if current_session() is None:
            current_session(self)

    def add_stderr_logger(self, level=logging.INFO):
        handler = logging.StreamHandler()
        handler.setLevel(level=level)
        self.message_log.addHandler(handler)
        self.message_log.setLevel(level)
        return handler

    @property
    def username(self):
        return self._settings.get('username')

    @property
    def hostname(self):
        return self._settings.get('domain')


    def send(self, request, **kwargs):
        if self.message_log.isEnabledFor(logging.DEBUG):
            r = copy.deepcopy(request)
            for filter in self.filters:
                r = filter(r)

            self.message_log.debug(
                'HTTP/1.1 {verb} {url}\n{headers}\nBody:\n{body}'.format(
                    verb=r.method,
                    url=r.url,
                    headers='\n'.join(
                        '{}: {}'.format(k, v) for k, v in r.headers.items()),
                    body=pformat(r.body)))
        else:
            self.message_log.info('HTTP/1.1 %s %s', request.method,
                                  request.url)

        response = super(Session, self).send(request, **kwargs)

        if self.message_log.isEnabledFor(logging.DEBUG):
            r = copy.deepcopy(response)
            for filter in self.filters:
                r = filter(r)

            self.message_log.debug(
                'HTTP {status} {url}\n{headers}\nBody:\n{body}'.format(
                    status=r.status_code,
                    url=r.url,
                    headers='\n'.join(
                        '{}: {}'.format(k, v) for k, v in r.headers.items()),
                    body=pformat(r.text)))
        else:
            self.message_log.info('HTTP/1.1 %s %s', response.status_code,
                                  response.url)

        return response

    def request(self, method, url,
                params=None, data=None, headers=None, cookies=None, files=None,
                auth=None, timeout=None, allow_redirects=True, proxies=None,
                hooks=None, stream=None, verify=None, cert=None, json=None):

        url = self._build_url(url)

        try:
            return super(Session, self).request(method, url, params, data,
                                                headers, cookies, files, auth,
                                                timeout, allow_redirects,
                                                proxies, hooks, stream, verify,
                                                cert, json)
        except requests.exceptions.SSLError as e:
            if 'REQUESTS_CA_BUNDLE' not in os.environ:
                raise RuntimeError(
                    "SSL handshake failed.  The 'REQUESTS_CA_BUNDLE' "
                    "environment variable should contain the path to the CA "
                    "certificate.  Alternatively, set verify_ssl=False to "
                    "disable certificate verification.")
            else:
                raise e

    def get(self, url, **kwargs):
        return self.request('GET', url, **kwargs)

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)

    def put(self, url, **kwargs):
        return self.request('PUT', url, **kwargs)

    def head(self, url, **kwargs):
        return self.request('HEAD', url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request('DELETE', url, **kwargs)

    def _get_token_with_kerberos(self):
        """Authenticate with a Kerberos ticket."""
        if kerberos is None:
            raise RuntimeError("Kerberos package not found.  Run 'pip "
                               "install sasctl[kerberos]' to install.")

        user = self._settings.get('username')
        # realm = user.rsplit('@', maxsplit=1)[-1] if '@' in user else None
        client_id = 'sas.tkmtrb'
        flags = kerberos.GSS_C_MUTUAL_FLAG | kerberos.GSS_C_SEQUENCE_FLAG
        service = 'HTTP@%s' % self._settings['domain']

        logger.info('Attempting Kerberos authentication to %s as %s'
                    % (service, user))

        url = self._build_url(
            '/SASLogon/oauth/authorize?client_id=%s&response_type=token'
            % client_id)

        # Get Kerberos challenge
        r = self.get(url, allow_redirects=False, verify=self.verify)

        if r.status_code != 401:
            raise ValueError('Kerberos challenge response not received.  '
                             'Expected HTTP 401 but received %s' %
                             r.status_code)

        if 'www-authenticate' not in r.headers:
            raise ValueError("Kerberos challenge response not received.  "
                             "'WWW-Authenticate' header not received.")

        if 'Negotiate' not in r.headers['www-authenticate']:
            raise ValueError("Kerberos challenge response not received.  "
                             "'WWW-Authenticate' header contained '%s', "
                             "expected 'Negotiate'."
                             % r.headers['www-authenticate'])

        # Initialize a request to KDC for a ticket to access the service.
        _, context = kerberos.authGSSClientInit(service,
                                                principal=user,
                                                gssflags=flags)

        # Send the request.
        # NOTE: empty-string parameter required for initial call.
        kerberos.authGSSClientStep(context, '')

        # Get the KDC response
        auth_header = 'Negotiate %s' % kerberos.authGSSClientResponse(context)

        # Get the user that was used for authentication
        username = kerberos.authGSSClientUserName(context)
        logger.info('Authenticated as %s' % username)

        # Drop @REALM from username and store
        if username is not None:
            self._settings['username'] = username.rsplit('@', maxsplit=1)[0]



        # Response to Kerberos challenge with ticket
        r = self.get(url,
                     headers={'Authorization': auth_header},
                     allow_redirects=False,
                     verify=self.verify)

        if 'Location' not in r.headers:
            raise ValueError("Invalid authentication response."
                             "'Location' header not received.")

        match = re.search('(?<=access_token=)[^&]*', r.headers['Location'])

        if match is None:
            raise ValueError("Invalid authentication response.  'Location' "
                             "header does not contain an access token.")

        return match.group(0)

    def _get_token_with_password(self):
        """Authenticate with a username and password."""
        username = self._settings['username']
        password = self._settings['password']
        url = self._build_url('/SASLogon/oauth/token')

        data = 'grant_type=password&username={}&password={}'.format(username,
                                                                    password)
        headers = {'Accept': 'application/json',
                   'Content-Type': 'application/x-www-form-urlencoded'}

        r = super(Session, self).post(url,
                                      data=data,
                                      headers=headers,
                                      auth=('sas.ec', ''))

        if r.status_code == 401:
            raise exceptions.AuthenticationError(username)
        else:
            r.raise_for_status()

        return r.json().get('access_token')

    def get_token(self):
        """Authenticates with the session host and retrieves an
        authorization token for use by subsequent requests.

        Returns
        -------
        str
            a bearer token for :class:`HTTPBearerAuth`

        Raises
        ------
        AuthenticationError
            authentication with the host failed

        """
        username = self._settings['username']
        password = self._settings['password']

        if username is None or password is None:
            return self._get_token_with_kerberos()
        else:
            return self._get_token_with_password()

    def _build_url(self, url):
        """Build a complete URL from a path by substituting in session parameters."""
        components = urlsplit(url)

        domain = components.netloc or self._settings['domain']

        # Add port if a non-standard port was specified
        if self._settings['port'] is not None and ':' not in domain:
            domain = '{}:{}'.format(domain, self._settings['port'])

        return urlunsplit([
            components.scheme or self._settings['protocol'],
            domain,
            components.path,
            components.query,
            components.fragment
        ])

    def __enter__(self):
        super(Session, self).__enter__()

        # Make this the current session
        global _session
        self._old_session = _session
        _session = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous current session
        global _session
        _session = self._old_session

        super(Session, self).__exit__()

    def __str__(self):
        return ("{class_}(hostname='{hostname}', username='{username}', "
                "protocol='{protocol}', verify_ssl={verify})".format(
            class_=type(self).__name__,
            hostname=self.hostname,
            username=self.username,
            protocol=self._settings.get('protocol'),
            verify=self.verify))

def is_uuid(id):
    try:
        UUID(str(id))
        return True
    except (ValueError, TypeError):
        return False


def get(*args, **kwargs):
    try:
        return request('get', *args, **kwargs)
    except HTTPError as e:
        if e.code == 404:
            return None  # Resource not found
        else:
            raise e


def head(*args, **kwargs):
    return request('head', *args, **kwargs)


def post(*args, **kwargs):
    return request('post', *args, **kwargs)


def put(*args, **kwargs):
    return request('put', *args, **kwargs)


def delete(*args, **kwargs):
    return request('delete', *args, **kwargs)


def request(verb, path, session=None, raw=False, **kwargs):
    session = session or current_session()

    if session is None:
        raise TypeError('No `Session` instance found.')

    response = session.request(verb, path, **kwargs)

    if 400 <= response.status_code <= 599:
        raise HTTPError(response.url, response.status_code, response.text,
                        response.headers, None)

    try:
        if raw:
            return response.json()
        else:
            obj = _unwrap(response.json())

            # ETag is required to update any object
            # May not be returned on all responses (e.g. listing multiple objects)
            if isinstance(obj, RestObj):
                setattr(obj, '_headers', response.headers)
            return obj
    except ValueError:
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
    if isinstance(obj, dict) and 'links' in obj:
        if isinstance(obj['links'], dict):
            return obj['links'].get(rel)
        else:
            links = [l for l in obj.get('links', []) if l.get('rel') == rel]
            if len(links) == 1:
                return links[0]
            elif len(links) > 1:
                return links


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


    return request(link['method'], link['href'], **kwargs)


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
        link = get_link(obj, 'self')
        if isinstance(link, dict):
            return link.get('uri')


def _unwrap(json):
    """Converts a JSON response to one or more `RestObj` instances.

    If the JSON contains a .items property, only those items are converted and returned.

    Parameters
    ----------
    json

    Returns
    -------

    """
    if 'items' in json:
        if len(json['items']) == 1:
            return RestObj(json['items'][0])
        elif len(json['items']) > 1:
            return list(map(RestObj, json['items']))
        else:
            return []
    return RestObj(json)


def _build_crud_funcs(path, single_term=None, plural_term=None,
                      service_name=None):
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

    @sasctl_command('list')
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
        params = 'filter={}'.format(filter) if filter is not None else {}

        results = get(path, params=params)
        return results if isinstance(results, list) else [results]

    @sasctl_command('get')
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
        if isinstance(item, dict) and all([k in item for k in ('id', 'name')]):
            if refresh:
                item = item['id']
            else:
                return item

        if is_uuid(item):
            return get(path + '/{id}'.format(id=item))
        else:
            results = list_items(filter='eq(name, "{}")'.format(item))

            # Not sure why, but as of 19w04 the filter doesn't seem to work.
            for result in results:
                if result['name'] == str(item):
                    # Make a request for the specific object so that ETag is included, allowing updates.
                    if get_link(result, 'self'):
                        return request_link(result, 'self')
                    else:
                        id = result.get('id', result['name'])
                    return get(path + '/{id}'.format(id=id))

            return None

        assert item is None or isinstance(item, dict)
        return item

    @sasctl_command('update')
    def update_item(item):
        """Updates a {item} instance.

        Parameters
        ----------
        item : dict

        Returns
        -------
        None

        """

        headers = getattr(item, '_headers', None)
        if headers is None or headers.get('etag') is None:
            raise ValueError('Could not find ETag for update of %s.' % item)

        id = getattr(item, 'id', None)
        if id is None:
            raise ValueError(
                'Could not find property `id` for update of %s.' % item)

        headers = {'If-Match': item._headers.get('etag'),
                   'Content-Type': item._headers.get('content-type')}

        return put(path + '/%s' % id, json=item, headers=headers)

    @sasctl_command('delete')
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
        if not (isinstance(item, dict) and 'id' in item):
            item = get_item(item)

        if isinstance(item, dict) and 'id' in item:
            item = item['id']

        if is_uuid(item):
            return delete(path + '/{id}'.format(id=item))
        else:
            raise ValueError("Unrecognized id '%s'" % item)

    # Pull object name from path if unspecified (many paths end in /folders or /repositories).
    plural_term = plural_term or str(path).split('/')[-1]
    single_term = single_term or plural_term
    service_name = service_name or plural_term
    service_name = service_name.replace(' ', '_')

    for func in [list_items, get_item, update_item, delete_item]:
        func.__doc__ = func.__doc__.format(item=single_term, items=plural_term)
        func._cli_service = service_name

        prefix = func.__name__.split('_')[0] + '_'
        suffix = plural_term if prefix == 'list_' else single_term
        func.__name__ = prefix + suffix

    return list_items, get_item, update_item, delete_item


def _build_is_available_func(service_root):
    def is_available():
        """Checks if the service is currently available.

        Returns
        -------
        bool

        """
        response = current_session().head(service_root + '/')
        return response.status_code == 200
    return is_available


class SasctlError(Exception):
    pass

class TimeoutError(SasctlError):
    pass
