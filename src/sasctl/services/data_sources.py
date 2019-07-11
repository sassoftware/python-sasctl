#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enables retrieval of data source metadata.

The Data Sources API works in concert with the Data Tables and Row Sets APIs
to navigate, reference, and retrieve data in the SAS Viya ecosystem. The
Data Sources API enables retrieval of metadata for data sources and linking
to their respective tables.

"""

from ..core import current_session, get, get_link, request_link, \
    _build_crud_funcs

_SERVICE_ROOT = '/dataSources'


def is_available():
    """Checks if the service is currently available.

    Returns
    -------
    bool

    """
    response = current_session().head(_SERVICE_ROOT + '/')
    return response.status_code == 200


def info():
    """Version and build information for the service.

    Returns
    -------
    RestObj

    """
    return get(_SERVICE_ROOT + '/apiMeta')


list_providers, \
_, _, _ = _build_crud_funcs(_SERVICE_ROOT + '/providers', 'provider')


def get_provider(provider, refresh=False):
    """Returns a provider instance.

    Parameters
    ----------
    provider : str or dict
        Name, ID, or dictionary representation of the provider.
    refresh : bool, optional
        Obtain an updated copy of the {item}.

    Returns
    -------
    RestObj or None
        A dictionary containing the provider attributes or None.

    Notes
    -------
    If `provider` is a complete representation of the provider it will be
    returned unless `refresh` is set.  This prevents unnecessary REST calls
    when data is already available on the client.

    """
    if isinstance(provider, dict) and 'id' in provider:
        if refresh:
            provider = provider['id']
        else:
            return provider

    return get(_SERVICE_ROOT + '/providers/{id}'.format(id=provider))


def list_sources(provider):
    """List all data sources available for a provider.

    Parameters
    ----------
    provider : str or dict
        Name, ID, or dictionary representation of the provider.

    Returns
    -------
    list
        A collection of :class:`.RestObj` instances.

    """

    provider = get_provider(provider)

    sources = request_link(provider, 'dataSources')
    if isinstance(sources, list):
        return sources
    else:
        return [sources]


def get_source(provider, source):
    """Returns a data source belonging to a given provider.

    Parameters
    ----------
    provider : str or dict
        Name, ID, or dictionary representation of the provider.
    source : str
        Name or id of the data source

    Returns
    -------
    RestObj or None
        A dictionary containing the data source attributes or None.

    """
    if isinstance(source, dict) and 'providerId' in source:
        return source

    sources = list_sources(provider)

    for s in sources:
        if source in (s.name, s.id):
            return s


def list_caslibs(source='cas-shared-default', filter=None):
    """Get all caslibs registered with the given CAS server.

    Parameters
    ----------
    source : str, optional
        Name of the CAS server.  Defaults to `cas-shared-default`.
    filter : str, optional

    Returns
    -------
    list
        A collection of :class:`.RestObj` instances.

    Notes
    -----
    See the filtering_ reference for details on the `filter` parameter.

    .. _filtering: https://developer.sas.com/reference/filtering/

    """
    source = get_source('cas', source)

    params = 'filter={}'.format(filter) if filter is not None else {}
    result = request_link(source, 'children', params=params)

    return result if isinstance(result, list) else [result]


def get_caslib(name, source=None):
    """Get a caslib by name.

    Parameters
    ----------
    name : str
        Name of the caslib
    source : str, optional
        Name of the CAS server.  Defaults to `cas-shared-default`.

    Returns
    -------
    RestObj

    """
    source = source or 'cas-shared-default'
    caslibs = list_caslibs(source, filter='eq(name, "%s")' % name)

    # caslibs = [c for c in caslibs if c.name == name]

    if len(caslibs):
        return caslibs.pop()


def list_tables(caslib, filter=None):
    """List tables available in a caslib.

    Parameters
    ----------
    caslib : str or dict
        Name, ID, or dictionary representation of the caslib.
    filter : str, optional

    Returns
    -------
    list
        A collection of :class:`.RestObj` instances.

    Notes
    -----
    See the filtering_ reference for details on the `filter` parameter.

    .. _filtering: https://developer.sas.com/reference/filtering/

    """
    if not get_link(caslib, 'tables'):
        caslib = get_caslib(caslib)

    params = 'filter={}'.format(filter) if filter is not None else {}
    result = request_link(caslib, 'tables', params=params)

    return result if isinstance(result, list) else [result]



def get_table(name, caslib, server=None):
    """Get metadata for a CAS table.

    Parameters
    ----------
    name : str
        Name of the table
    caslib : str or dict
        Name, ID, or dictionary representation of the caslib.
    server : str
        Name of the CAS server on which the `caslib` is registered.

    Returns
    -------
    RestObj

    """
    tables = list_tables(caslib, filter='eq(name, "%s")' % name)

    if len(tables):
        return tables.pop()




