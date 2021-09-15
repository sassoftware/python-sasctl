#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from .service import Service

DEFAULT_SERVER = 'cas-shared-default'
DEFAULT_CASLIB = 'casuser'


class CASManagement(Service):
    """The CAS Management service provides the ability to manage and perform
    actions on common resources as they relate to Cloud Analytic Services (CAS).
    """

    _SERVICE_ROOT = '/casManagement'

    list_servers, get_server, _, _ = Service._crud_funcs('/servers', 'server')

    @classmethod
    def list_caslibs(cls, server, filter_=None):
        """List caslibs available on a server.

        Parameters
        ----------
        server : str or dict
            Name, ID, or dictionary representation of the server.
        filter_ : str, optional
            A `formatted <https://developer.sas.com/reference/filtering>`_
            filter string.

        Returns
        -------
        list
            A collection of :class:`.RestObj` instances.

        """
        return (
            cls._get_rel(server, 'caslibs', func=cls.get_server, filter_=filter_) or []
        )

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
        server = server or DEFAULT_SERVER
        caslibs = cls.list_caslibs(server, filter_='eq($primary,name, "%s")' % name)

        if caslibs:
            return caslibs[0]
        return None

    @classmethod
    def list_tables(cls, caslib, server=None, filter_=None):
        """List tables available in a caslib.

        Parameters
        ----------
        caslib : str or dict
            Name, ID, or dictionary representation of the caslib.
        server : str, optional
            Server where the `caslib` is registered.
        filter_ : str, optional
            Filter string in the `https://developer.sas.com/reference/filtering
            /` format.

        Returns
        -------
        list
            A collection of :class:`.RestObj` instances.

        """
        return cls._get_rel(caslib, 'tables', server, func=cls.get_caslib, filter_=filter_) or []

    @classmethod
    def get_table(cls, name, caslib=None, server=None):
        """Get a table by name.

        Parameters
        ----------
        name : str
            Name of the table.
        caslib : str or dict, optional
            Name, ID, or dictionary representation of the caslib.  Defaults to
            CASUSER.
        server : str, optional
            Server where the `caslib` is registered.

        Returns
        -------
        RestObj

        """
        caslib = caslib or DEFAULT_CASLIB
        tables = cls.list_tables(
            caslib, server=server, filter_='eq($primary,name, "%s")' % name
        )

        if tables:
            return tables[0]
        return None

    @classmethod
    def upload_file(
        cls, file, name, caslib=None, server=None, header=None, format_=None
    ):
        """Upload a file to a CAS table.

        Uploads the contents of a CSV, XLS, XLSX, SAS7BDT or SASHDAT file to a
        newly created CAS table.

        Parameters
        ----------
        file : str or file-like object
            File containing data to upload or path to the file.
        name : str
            Name of the table to create
        caslib : str, optional
            caslib in which the table will be created.  Defaults to CASUSER.
        server : str, optional
            CAS server on which the table will be created.  Defaults to
            cas-shared-default.
        header : bool, optional
            Whether the first row of data contains column headers.  Defaults to
            True.
        format_ : {"csv", "xls", "xlsx", "sas7bdat", "sashdat"}, optional
            File of input `file`.  Not required if format can be discerned from
            the file path.

        Returns
        -------
        RestObj
            Table reference

        """
        name = str(name)
        caslib = caslib or DEFAULT_CASLIB
        server = server or DEFAULT_SERVER
        header = True if header is None else bool(header)

        # Not a file-like object, assuming it's a file path
        if not hasattr(file, 'read'):
            path = os.path.abspath(os.path.expanduser(file))
            format_ = os.path.splitext(path)[-1].lstrip('.').lower()

            # Extension should be supported & needs to be explicitly set in
            # the "format" parameter to avoid errors.
            if format_ not in ('csv', 'xls', 'xlsx', 'sas7bdat', 'sashdat'):
                raise ValueError("File '%s' has an unsupported file type." % file)

            with open(path, 'rb') as f:
                file = f.read()

        data = {
            'tableName': name,
            'containsHeaderRow': header,
        }

        if format_ is not None:
            data['format'] = format_

        tbl = cls.post(
            '/servers/%s/caslibs/%s/tables' % (server, caslib),
            data=data,
            files={name: file},
        )
        return tbl
