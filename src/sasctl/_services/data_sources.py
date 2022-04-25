#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..core import PagedItemIterator
from .service import Service
from .cas_management import DEFAULT_CASLIB


class DataSources(Service):
    """Enables retrieval of data source metadata.

    The Data Sources API works in concert with the Data Tables and Row Sets
    APIs to navigate, reference, and retrieve data in the SAS Viya ecosystem.
    The Data Sources API enables retrieval of metadata for data sources and
    linking to their respective tables.

    """

    _SERVICE_ROOT = "/dataSources"

    list_providers, _, _, _ = Service._crud_funcs("/providers", "provider")

    @classmethod
    def get_provider(cls, provider, refresh=False):
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
        if isinstance(provider, dict) and "id" in provider:
            if refresh:
                provider = provider["id"]
            else:
                return provider

        return cls.get("/providers/{id}".format(id=provider))

    @classmethod
    def list_sources(cls, provider):
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

        provider = cls.get_provider(provider)

        sources = cls.request_link(provider, "dataSources")
        if isinstance(sources, (list, PagedItemIterator)):
            return sources
        return [sources]

    @classmethod
    def get_source(cls, provider, source):
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
        if isinstance(source, dict) and "providerId" in source:
            return source

        sources = cls.list_sources(provider)

        for s in sources:
            if source in (s.name, s.id):
                return s
        return None

    @classmethod
    def list_caslibs(cls, source="cas-shared-default", filter_=None):
        """Get all caslibs registered with the given CAS server.

        Parameters
        ----------
        source : str, optional
            Name of the CAS server.  Defaults to `cas-shared-default`.
        filter_ : str, optional

        Returns
        -------
        list
            A collection of :class:`.RestObj` instances.

        Notes
        -----
        See the filtering_ reference for details on the `filter` parameter.

        .. _filtering: https://developer.sas.com/reference/filtering/

        """
        source = cls.get_source("cas", source)

        params = "filter={}".format(filter_) if filter_ is not None else {}
        result = cls.request_link(source, "children", params=params)

        return result if isinstance(result, (list, PagedItemIterator)) else [result]

    @classmethod
    def get_caslib(cls, name, source=None):
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
        source = source or "cas-shared-default"
        caslibs = cls.list_caslibs(source, filter_='eq(name, "%s")' % name)

        if caslibs:
            return caslibs[0]
        return None

    @classmethod
    def list_tables(cls, caslib, filter_=None, session_id=None):
        """List tables available in a caslib.

        Parameters
        ----------
        caslib : str or dict
            Name, ID, or dictionary representation of the caslib.
        filter_ : str, optional
        session_id : str, optional
            ID of an existing CAS session to use.  Otherwise a new session will
            be created.

        Returns
        -------
        list
            A collection of :class:`.RestObj` instances.

        Notes
        -----
        See the filtering_ reference for details on the `filter` parameter.

        .. _filtering: https://developer.sas.com/reference/filtering/

        """
        if not cls.get_link(caslib, "tables"):
            caslib = cls.get_caslib(caslib)

        params = {}

        if filter_ is not None:
            params["filter"] = str(filter_)
        if session_id is not None:
            params["sessionId"] = str(session_id)

        params = "&".join("%s=%s" % (k, params[k]) for k in params)

        result = cls.request_link(caslib, "tables", params=params)

        return result if isinstance(result, (list, PagedItemIterator)) else [result]

    @classmethod
    def get_table(cls, name, caslib, session_id=None):
        """Get metadata for a CAS table.

        Parameters
        ----------
        name : str
            Name of the table
        caslib : str or dict
            Name, ID, or dictionary representation of the caslib.

        Returns
        -------
        RestObj

        """
        tables = cls.list_tables(
            caslib, filter_='eq(name, "%s")' % name, session_id=session_id
        )

        if tables:
            return tables[0]
        return None

    @classmethod
    def table_uri(cls, table):
        """Get the URI identifying a table.

        Parameters
        ----------
        table : dict or CASTable

        Returns
        -------
        str
            table URI

        """

        # Use the CASTable information to find the same table
        if all(hasattr(table, x) for x in ("name", "caslib", "get_connection")):
            from .cas_management import CASManagement as cm

            server_http = table.builtins.httpAddress()
            server_name = server_http["restPrefix"].lstrip("/").rsplit("-http")[0]

            table_name = table.name
            caslib_name = table.caslib

            # If caslib is not set, empty dict is returned.
            # Build the default caslib name (data_sources and cas_management
            # services currently require username to be explicitly included)
            if caslib_name == {}:
                # skipcq: PYL-W0212
                username = table.get_connection()._username
                caslib_name = DEFAULT_CASLIB + "(%s)" % username

            # session_id = table.get_connection()._session

            caslib = cm.get_caslib(caslib_name, server_name)
            table = cm.get_table(table_name, caslib)

        # Responses directly from cas_management service have a URI to
        # data_tables service under the "dataTable" rel
        link = cls.get_link(table, "dataTable")
        if link is not None:
            return link["uri"]

        # Responses from data_tables service have the URI to data_tables under
        # the "self" rel
        link = cls.get_link(table, "self")
        if link is not None:
            return link["uri"]

        # Responses from upload operations often don't include a links section.
        # Fall back to tableReference section if present
        table_ref = table.get("tableReference", {})
        return table_ref.get("tableUri")
