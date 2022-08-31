#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from .service import Service


from typing import Union, TextIO


DEFAULT_SERVER = "cas-shared-default"
DEFAULT_CASLIB = "casuser"


class CASManagement(Service):
    """The CAS Management service provides the ability to manage and perform
    actions on common resources as they relate to Cloud Analytic Services (CAS).
    """

    _SERVICE_ROOT = "/casManagement"

    list_servers, get_server, _, _ = Service._crud_funcs("/servers", "server")

    @classmethod
    def list_sessions(cls, qparams: dict = None, server: str = None):
        """
        Returns a collection of sessions available on the CAS server.

        Params
        ------
        queryParams : dict, optional
            Query parameters.
            Valid keys are `start`, `limit`, `filter`,
            `sortBy`, `excludeItemLink`, `sessionId`.
        server : str, optional
            Name of the CAS server. Defaults to 'cas-shared-default'.

        Returns
        -------
        list
            A collection of :class:`.RestObj` instances.
        """
        server = server or DEFAULT_SERVER

        if qparams is not None:
            allowedQuery = [
                "start",
                "limit",
                "filter",
                "sortBy",
                "excludeItemLink",
                "sessionId",
            ]

            if not all(key in allowedQuery for key in qparams.keys()):
                raise ValueError(
                    "The only acceptable queries are %s." % (allowedQuery)
                )
            else:
                query = qparams
        else:
            query={}
        
        sesslist = cls.get(
            "/servers/%s/sessions"  % (server),
            params = query
            )
        return sesslist

    @classmethod
    def create_session(cls, properties: dict, server: str = None):
        """Creates a new session on the CAS server.

        Params
        ------
        properties : dict
            Properties of the session.
            Valid keys are `authenticationType` (required),
            `locale`, `name`, `nodeCount`, `timeOut`.
        server : str
            Name of the CAS server.  Defaults to `cas-shared-default`.

        Returns
        -------
        RestObj
        """
        server = server or DEFAULT_SERVER

        allowedBodyKeys = [
            "authenticationType",
            "locale",
            "name",
            "nodeCount",
            "replace",
            "timeOut",
        ]

        if not all(key in allowedBodyKeys for key in properties.keys()):
            raise ValueError(
                "The only acceptable properties are %s." % (allowedBodyKeys)
            )

        if "authenticationType" not in properties.keys():
            raise ValueError("The property 'authenticationType' is required.")

        sess = cls.post("/servers/%s/sessions" % (server), json=properties)
        return sess

    @classmethod
    def delete_session(cls, sess_id: str, server: str = None, qparams: dict = None):
        """Terminates a session on the CAS server.

        Params
        ------
        sess_id : str
            A string indicating the Session id.
        server : str
            Name of the CAS server.  Defaults to `cas-shared-default`.
        qparams : dict, optional
            Query parameters.
            Valid keys are `force`, `superUserSessionId`.

        Returns
        -------
        RestObj
        """
        server = server or DEFAULT_SERVER

        if qparams is not None:
            allowedQueryKeys = ["force", "superUserSessionId"]

            if not all(key in allowedQueryKeys for key in qparams.keys()):
                raise ValueError(
                    "The only acceptable queries are %s." % (allowedQueryKeys)
                )
        else:
            qparams = {}

        sess = cls.delete("/servers/%s/sessions/%s" % (server, sess_id), params=qparams)
        return sess

    @classmethod
    def list_caslibs(cls, server: Union[str, dict], filter_: str = None):
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
            cls._get_rel(server, "caslibs", func=cls.get_server, filter_=filter_) or []
        )

    @classmethod
    def get_caslib(cls, name: str, server: str = None):
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
    def list_tables(
        cls, caslib: Union[str, dict], server: str = None, filter_: str = None
    ):
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
        return (
            cls._get_rel(caslib, "tables", server, func=cls.get_caslib, filter_=filter_)
            or []
        )

    @classmethod
    def get_table(cls, name: str, caslib: Union[str, dict] = None, server: str = None):
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
        cls,
        file: Union[str, TextIO],
        name: str,
        caslib: str = None,
        server: str = None,
        header: bool = None,
        format_: str = None,
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
        if not hasattr(file, "read"):
            path = os.path.abspath(os.path.expanduser(file))
            format_ = os.path.splitext(path)[-1].lstrip(".").lower()

            # Extension should be supported & needs to be explicitly set in
            # the "format" parameter to avoid errors.
            if format_ not in ("csv", "xls", "xlsx", "sas7bdat", "sashdat"):
                raise ValueError("File '%s' has an unsupported file type." % file)

            with open(path, "rb") as f:
                file = f.read()

        data = {
            "tableName": name,
            "containsHeaderRow": header,
        }

        if format_ is not None:
            data["format"] = format_

        tbl = cls.post(
            "/servers/%s/caslibs/%s/tables" % (server, caslib),
            data=data,
            files={name: file},
        )
        return tbl

    @classmethod
    def update_state_table(
        cls,
        value: str,
        name: str,
        caslib: str = None,
        server: str = None,
        *,
        qparams: dict = None,
        body: dict = dict()
    ):
        """Modifies the state of a table to loaded or unloaded.
        Returns loaded or unloaded to indicate the state after the operation.

        Parameters
        ----------
        value : str, required
            State to which to set the table. Valid values include `loaded` or `unloaded`.
        name : str, required
            Name of the table.
        caslib : str, optional
            Name of the caslib. Defaults to `CASUSER`.
        server : str, optional
            Server where the `caslib` is registered.
            Defaults to `cas-shared-default`.
        qparams: dict, optional
            Additional query parameters.
            Valid keys are `sessionId`, `scope`, `sourceTableName`, `createRelationships`
        body : dict, optional
            Extra instructions providing greater control over the output when a state change to loaded is requested.
            Valid keys are `copies`, `label`, `outputCaslibName`, `outputTableName`, `parameters`,
            `replace`, `replaceMode`, `scope`.

        Returns
        -------
        RestObj

        """
        server = server or DEFAULT_SERVER
        caslib = caslib or DEFAULT_CASLIB

        if value in ["loaded", "unloaded"]:
            query = {"value": value}
        else:
            raise ValueError(
                "The state can only have values of `loaded` or `unloaded`."
            )

        if qparams is not None:
            allowedQueryKeys = [
                "sessionId",
                "scope",
                "sourceTableName",
                "createRelationships",
            ]

            if not all(key in allowedQueryKeys for key in qparams.keys()):
                raise ValueError(
                    "The only query parameters allowed are %s." % (allowedQueryKeys)
                )
            else:
                query.update(qparams)

        allowedBodyKeys = [
            "copies",
            "label",
            "outputCaslibName",
            "outputTableName",
            "parameters",
            "replace",
            "replaceMode",
            "scope",
        ]

        if not all(key in allowedBodyKeys for key in body.keys()):
            raise ValueError(
                "The body accepts only the following parameters %s." % (allowedBodyKeys)
            )

        tbl = cls.put(
            "/servers/%s/caslibs/%s/tables/%s/state" % (server, caslib, name),
            params=query,
            json=body,
        )

        return tbl

    @classmethod
    def promote_table(cls, name: str, sessId: str, caslib: str, server: str = None):
        """Changes the scope of a loaded CAS table from session to global scope.
        Operation valid only on a session table.
        Promote target is the same Caslib that contains the session table.

        Parameters
        ----------
        name : str
            Name of the table.
        sessId: str
            The session ID
        caslib : str
            Name of the caslib.
        server : str
            Server where the `caslib` is registered.
            Defaults to `cas-shared-default`.

        Returns
        -------
        RestObj

        """
        server = server or DEFAULT_SERVER

        query = {"value": "global", "sessionId": sessId}

        tbl = cls.put(
            "/servers/%s/caslibs/%s/tables/%s/scope" % (server, caslib, name),
            params=query,
        )

        return tbl

    @classmethod
    def save_table(
        cls,
        name: str,
        caslib: str,
        properties: dict = None,
        sessId: str = None,
        server: str = None,
    ):
        """Saves a CAS table to a source table

        Parameters
        ----------
        name : str
            Name of the table.
        sessId: str
            The session ID
        caslib : str
            Name of the caslib.
        propertes : dict, optional
            Properties of the table.
            Valid keys are `caslibName`, `format`, `replace`,
            `compress`, `tableName`, `sourceTableName`, `parameters`.
        server : str
            Server where the `caslib` is registered. Defaults to `cas-shared-default`.

        Returns
        -------
        RestObj
        """
        server = server or DEFAULT_SERVER

        if properties:
            allowedBodyKeys = [
                "caslibName",
                "format",
                "replace",
                "compress",
                "tableName",
                "sourceTableName",
                "parameters",
            ]

            if not all(key in allowedBodyKeys for key in properties.keys()):
                raise ValueError(
                    "The only acceptable properties are %s." % (allowedBodyKeys)
                )
        else:
            properties = {}

        query = {"sessionId": sessId} if sessId else {}

        sess = cls.post(
            "/servers/%s/caslibs/%s/tables/%s" % (server, caslib, name),
            params=query,
            json=properties,
        )
        return sess
    
    @classmethod
    def del_table(
        cls, 
        name: str, 
        qParam: dict = None, 
        caslib: str = None,
        server: str = None 
    ):
        """Deletes a table from Caslib source. Note that is not an unload.
        This operation physically removes the source table (if the source is writable).
        For path-based caslibs, this physically removes the file.    
        
        Parameters
        ----------
        name : str
            Name of the table.
        qParam : dict
            Query parameters. Note that some are required.
            The allowed query parameters are `sessionId`, 
            `sourceTableName`, `quiet`, `removeAcs`.
        caslib : str
            Name of the caslib. Defaults to 'CASUSER'
        server : str
            Server where the `caslib` is registered.
            Defaults to 'cas-shared-default'.

        Returns
        -------
        RestObj
        """
        
        server = server or DEFAULT_SERVER
        caslib = caslib or DEFAULT_CASLIB
        
        allowedQ = [
            "sessionId",
            "sourceTableName",
            "quiet",
            "removeAcs"
        ]

        if isinstance(qParam,dict) and all(key in allowedQ for key in qParam.keys()):
            if ("sourceTableName" not in qParam.keys() 
            and "quiet" not in qParam.keys() 
            and "removeAcs" not in qParam.keys()):
                raise Exception(
                    "Missing required query parameters `sourceTableName`, `quiet`, `removeAcs`"
                )
            else:
                query = qParam
        else:
            raise ValueError(
                "The only acceptable query parameters are %s and must be passed in a dictionary"
                % (allowedQ)
            )

        tbl =  cls.delete(
            "servers/%s/caslibs/%s/tables/%s"%(server,caslib,name),
            params=query
            )
        return tbl
