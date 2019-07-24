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

    def list_caslibs(self, server, filter=None):
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
        return self._get_rel(server, 'caslibs', self.get_server, filter) or []

    def get_caslib(self, name, server=None):
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
        caslibs = self.list_caslibs(server,
                                    filter='eq($primary,name, "%s")' % name)

        assert isinstance(caslibs, list)

        if len(caslibs):
            return caslibs.pop()

    def list_tables(self, caslib, server=None, filter=None):
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
        return self._get_rel(caslib, 'tables',
                             self.get_caslib, filter, server) or []


    def get_table(self, name, caslib, server=None):
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
        tables = self.list_tables(caslib,
                             server=server,
                             filter='eq($primary,name, "%s")' % name)

        if len(tables):
            return tables.pop()