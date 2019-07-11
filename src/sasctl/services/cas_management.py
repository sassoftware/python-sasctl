#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CAS Management service provides the ability to manage and perform
actions on common resources as they relate to Cloud Analytic Services (CAS).
"""

from ..core import request_link, get, _build_is_available_func, \
    _build_crud_funcs


_SERVICE_ROOT = '/casManagement'


is_available = _build_is_available_func(_SERVICE_ROOT)


def info():
    """Version and build information for the service.

    Returns
    -------
    RestObj

    """
    return get(_SERVICE_ROOT + '/apiMeta')


list_servers, get_server, _, _ = _build_crud_funcs(_SERVICE_ROOT + '/servers',
                                                   'server')


def _get_rel(item, rel, func=None, filter=None, *args):
    """Get `item` and request a link.

    Parameters
    ----------
    item : str or dict
    rel : str
    func : function, optional
        Callable that takes (item, *args) and returns a RestObj of `item`
    filter : str, optional

    args : any
        Passed to `func`

    Returns
    -------
    list

    """
    if func is not None:
        obj = func(item, *args)

    if obj is None:
        return

    params = 'filter={}'.format(filter) if filter is not None else {}

    resources = request_link(obj, rel, params=params)
    return resources if isinstance(resources, list) else [resources]


def list_caslibs(server, filter=None):
    """List caslibs available on a server.

    Parameters
    ----------
    server : str or dict
        Name, ID, or dictionary representation of the server.
    filter : str, optional
        A `formatted <https://developer.sas.com/reference/filtering>`_
        filter string.

    Returns
    -------
    list
        A collection of :class:`.RestObj` instances.

    """
    return _get_rel(server, 'caslibs', get_server, filter) or []


def get_caslib(name, server=None):
    """Get a caslib by name.

    Parameters
    ----------
    name : str
        Name of the caslib
    server : str, optional
        Name of the CAS server.  Defaults to `cas-shared-default`.

    Returns
    -------
    RestObj

    """
    server = server or 'cas-shared-default'
    caslibs = list_caslibs(server, filter='eq($primary,name, "%s")' % name)

    assert isinstance(caslibs, list)

    if len(caslibs):
        return caslibs.pop()


def list_tables(caslib, server=None, filter=None):
    """List tables available in a caslib.

    Parameters
    ----------
    caslib : str or dict
        Name, ID, or dictionary representation of the caslib.
    server : str, optional
        Server where the `caslib` is registered.
    filter : str, optional
        Filter string in the `https://developer.sas.com/reference/filtering
        /` format.

    Returns
    -------
    list
        A collection of :class:`.RestObj` instances.

    """
    return _get_rel(caslib, 'tables', get_caslib, filter, server) or []


def get_table(name, caslib, server=None):
    """Get a table by name.

    Parameters
    ----------
    name : str
        Name of the table.
    caslib : str or dict
        Name, ID, or dictionary representation of the caslib.
    server : str, optional
        Server where the `caslib` is registered.

    Returns
    -------
    RestObj

    """
    tables = list_tables(caslib,
                         server=server,
                         filter='eq($primary,name, "%s")' % name)

    if len(tables):
        return tables.pop()