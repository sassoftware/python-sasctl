#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service


class CASManagement(Service):
    """The CAS Management service provides the ability to manage and perform
    actions on common resources as they relate to Cloud Analytic Services (CAS).
    """

    _SERVICE_ROOT = '/casManagement'

    list_servers, get_server, _, _ = Service._crud_funcs('/servers', 'server')

    @classmethod
    def list_caslibs(cls, server, filter=None):
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
        return cls._get_rel(server, 'caslibs', func=cls.get_server,
                            filter=filter) or []

    @classmethod
    def get_caslib(cls, name, server=None):
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
        caslibs = cls.list_caslibs(server,
                                   filter='eq($primary,name, "%s")' % name)

        if len(caslibs) > 0:
            return caslibs.pop()

    @classmethod
    def list_tables(cls, caslib, server=None, filter=None):
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
        return cls._get_rel(caslib, 'tables', cls.get_caslib, filter,
                            server) or []

    @classmethod
    def get_table(cls, name, caslib, server=None):
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
        tables = cls.list_tables(caslib,
                                 server=server,
                                 filter='eq($primary,name, "%s")' % name)

        if len(tables) > 0:
            return tables.pop()