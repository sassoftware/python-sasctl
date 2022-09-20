#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Union, TextIO

from .service import Service

QUERY_PARAMETERS = "query parameters"
DEFAULT_SERVER = "cas-shared-default"
DEFAULT_CASLIB = "Public"


def check_keys(valid_keys: list, input_keys: list, parameters: str):
    """Compares the input_keys against the valid_keys
    to see if they are allowed to be passed
    as parameters in the request.

    Parameters
    ----------
    valid_keys: list
        List of allowed parameters
    input_keys: list
        List of input parameters
    parameters: str
        String describing the type of parameters
        that are being tested.

    Returns
    -------
    raises ValueError if input_keys are not valid
    """
    if not all(key in valid_keys for key in input_keys):
        raise ValueError(
            "The only acceptable values for %s are %s" % (parameters, valid_keys)
        )


def check_required_key(
    required_key: Union[str, list], input_keys: list, parameters: str
):
    """Check whether the required parameters
    are in the list of input_key.

    Parameters
    ----------
    required_key: str or list
        Required parameters
    input_keys: list
        The input parameters
    parameters: str
        String describing the type of parameters
        that are being tested.

    Returns
    -------
    raises ValueError if required_key is not present.
    raises TypeError if required_key is neither a list or a string.
    """
    if isinstance(required_key, str):
        if required_key not in input_keys:
            raise ValueError(
                "The %s is a required %s parameter." % (required_key, parameters)
            )
    elif isinstance(required_key, list):
        required_set = set(required_key)
        input_set = set(input_keys)
        if not required_set.issubset(input_set):
            raise ValueError(
                "The %s are required %s parameters." % (required_key, parameters)
            )
    else:
        raise TypeError(
            "Please enter either a list or a string of required parameters."
        )


class CASManagement(Service):
    """The CAS Management service provides the ability to manage and perform
    actions on common resources as they relate to Cloud Analytic Services (CAS).
    """

    _SERVICE_ROOT = "/casManagement"

    list_servers, get_server, _, _ = Service._crud_funcs("/servers", "server")

    @classmethod
    def list_sessions(cls, query_params: dict = None, server: str = None):
        """
        Returns a collection of sessions available on the CAS server.

        Parameters
        ------
        query_params : dict, optional
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

        if query_params is not None:
            allowed_query = [
                "start",
                "limit",
                "filter",
                "sortBy",
                "excludeItemLink",
                "sessionId",
            ]
            check_keys(allowed_query, query_params, QUERY_PARAMETERS)
        else:
            query_params = {}

        sess_list = cls.get("/servers/%s/sessions" % server, params=query_params)
        return sess_list

    @classmethod
    def create_session(cls, properties: dict, server: str = None):
        """Creates a new session on the CAS server.

        Parameters
        ------
        properties : dict
            Properties of the session.
            Valid keys are `authenticationType` (required),
            `locale`, `name`, `nodeCount`, `timeOut`.
        server : str
            Name of the CAS server.  Defaults to 'cas-shared-default'.

        Returns
        -------
        RestObj
        """
        server = server or DEFAULT_SERVER

        allowed_body = [
            "authenticationType",
            "locale",
            "name",
            "nodeCount",
            "replace",
            "timeOut",
        ]
        check_keys(allowed_body, properties.keys(), "body")
        check_required_key("authenticationType", properties.keys(), "body")

        sess = cls.post("/servers/%s/sessions" % (server), json=properties)
        return sess

    @classmethod
    def delete_session(
        cls, sess_id: str, server: str = None, query_params: dict = None
    ):
        """Terminates a session on the CAS server.

        Parameters
        ------
        sess_id : str
            A string indicating the Session id.
        server : str
            Name of the CAS server.  Defaults to 'cas-shared-default'.
        query_params : dict, optional
            Query parameters.
            Valid keys are `force`, `superUserSessionId`.

        Returns
        -------
        RestObj
        """
        server = server or DEFAULT_SERVER

        if query_params is not None:
            allowed_query = ["force", "superUserSessionId"]
            check_keys(allowed_query, query_params, QUERY_PARAMETERS)
        else:
            query_params = {}

        sess = cls.delete(
            "/servers/%s/sessions/%s" % (server, sess_id), params=query_params
        )
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
            Name of the CAS server. Defaults to 'cas-shared-default'.

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
        *,
        detail: dict = None
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
            caslib in which the table will be created. Defaults to 'Public'.
        server : str, optional
            CAS server on which the table will be created.  Defaults to
            cas-shared-default.
        header : bool, optional
            Whether the first row of data contains column headers.  Defaults to
            True.
        format_ : {"csv", "xls", "xlsx", "sas7bdat", "sashdat"}, optional
            File of input `file`.  Not required if format can be discerned from
            the file path.
        detail : dict, optional
            Additional body parameters. Allowed parameters are
            'sessionId', 'variables', 'label', 'scope', 'replace', 'encoding',
            'allowTruncation', 'allowEmbeddedNewLines', 'delimiter',
            'varchars', 'scanRows', 'threadCount', 'stripBlanks', 'sheetName',
            'password', 'decryptionKey', 'stringLengthMultiplier',
            'varcharConversionThreshold'.

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

        allowed_body = [
            "sessionId",
            "variables",
            "label",
            "scope",
            "replace",
            "encoding",
            "allowTruncation",
            "allowEmbeddedNewLines",
            "delimiter",
            "varchars",
            "scanRows",
            "threadCount",
            "stripBlanks",
            "sheetName",
            "password",
            "decryptionKey",
            "stringLengthMultiplier",
            "varcharConversionThreshold",
        ]
        allowed_csv = [
            "sessionId",
            "variables",
            "label",
            "scope",
            "replace",
            "encoding",
            "allowTruncation",
            "allowEmbeddedNewLines",
            "delimiter",
            "varchars",
            "scanRows",
            "threadCount",
            "stripBlanks",
        ]
        allowed_xls = [
            "sessionId",
            "variables",
            "label",
            "scope",
            "replace",
            "sheetName",
        ]
        allowed_sas = [
            "sessionId",
            "variables",
            "label",
            "scope",
            "replace",
            "password",
            "decryptionKey",
            "stringLengthMultiplier",
            "varcharConversionThreshold",
        ]

        if detail is not None:
            check_keys(allowed_body, detail.keys(), "body")
            if format_ == "csv":
                check_keys(allowed_csv, detail.keys(), "csv files")
            elif format_ in ["xls", "xlsx"]:
                check_keys(allowed_xls, detail.keys(), "excel files")
            elif format_ in ["sashdat", "sas7bdat"]:
                check_keys(allowed_sas, detail.keys(), "sas files")

            data.update(detail)

        tbl = cls.post(
            "/servers/%s/caslibs/%s/tables" % (server, caslib),
            data=data,
            files={"file": (name, file)},
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
        query_params: dict = None,
        body: dict = None
    ):
        """Modifies the state of a table to loaded or unloaded.
        Returns loaded or unloaded to indicate the state after the operation.

        Parameters
        ----------
        value : str
            State to which to set the table. Valid values include `loaded` or `unloaded`.
        name : str
            Name of the table.
        caslib : str, optional
            Name of the caslib. Defaults to 'Public'.
        server : str, optional
            Server where the `caslib` is registered.
            Defaults to 'cas-shared-default'.
        query_params: dict, optional
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

        if query_params is not None:
            allowed_query = [
                "sessionId",
                "scope",
                "sourceTableName",
                "createRelationships",
            ]
            check_keys(allowed_query, query_params.keys(), QUERY_PARAMETERS)
            query.update(query_params)

        if body is not None:
            allowed_body = [
                "copies",
                "label",
                "outputCaslibName",
                "outputTableName",
                "parameters",
                "replace",
                "replaceMode",
                "scope",
            ]
            check_keys(allowed_body, body.keys(), "body parameters")
        else:
            body = {}

        tbl = cls.put(
            "/servers/%s/caslibs/%s/tables/%s/state" % (server, caslib, name),
            params=query,
            json=body,
        )

        return tbl

    @classmethod
    def promote_table(cls, name: str, sess_id: str, caslib: str, server: str = None):
        """Changes the scope of a loaded CAS table from session to global scope.
        Operation valid only on a session table.
        Promote target is the same Caslib that contains the session table.

        Parameters
        ----------
        name : str
            Name of the table.
        sess_id: str
            The session ID
        caslib : str
            Name of the caslib.
        server : str
            Server where the `caslib` is registered.
            Defaults to 'cas-shared-default'.

        Returns
        -------
        RestObj

        """
        server = server or DEFAULT_SERVER

        query = {"value": "global", "sessionId": sess_id}

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
        sess_id: str = None,
        server: str = None,
    ):
        """Saves a CAS table to a source table

        Parameters
        ----------
        name : str
            Name of the table.
        sess_id: str
            The session ID
        caslib : str
            Name of the caslib.
        properties : dict, optional
            Properties of the table.
            Valid keys are `caslibName`, `format`, `replace`,
            `compress`, `tableName`, `sourceTableName`, `parameters`.
        server : str
            Server where the `caslib` is registered. Defaults to 'cas-shared-default'.

        Returns
        -------
        RestObj
        """
        server = server or DEFAULT_SERVER

        if properties is not None:
            allowed_body = [
                "caslibName",
                "format",
                "replace",
                "compress",
                "tableName",
                "sourceTableName",
                "parameters",
            ]
            check_keys(allowed_body, properties.keys(), "body")
        else:
            properties = {}

        query = {"sessionId": sess_id} if sess_id else {}

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
        query_params: dict = None,
        caslib: str = None,
        server: str = None,
    ):
        """Deletes a table from Caslib source. Note that is not an unload.
        This operation physically removes the source table (if the source is writable).
        For path-based caslibs, this physically removes the file.

        Parameters
        ----------
        name : str
            Name of the table.
        query_params : dict
            Query parameters.
            The allowed query parameters are `sessionId`,
            `sourceTableName`, `quiet`, `removeAcs`.
            Note that the last three are required.
        caslib : str
            Name of the caslib. Defaults to 'Public'
        server : str
            Server where the `caslib` is registered.
            Defaults to 'cas-shared-default'.

        Returns
        -------
        RestObj
        """

        server = server or DEFAULT_SERVER
        caslib = caslib or DEFAULT_CASLIB

        required_queries = ["sourceTableName", "quiet", "removeAcs"]

        if query_params is not None:
            allowed_query = ["sessionId", "sourceTableName", "quiet", "removeAcs"]
            check_keys(allowed_query, query_params.keys(), QUERY_PARAMETERS)
            check_required_key(required_queries, query_params.keys(), "query")
        else:
            raise ValueError(
                "You must provide these query parameters: %s" % required_queries
            )

        tbl = cls.delete(
            "servers/%s/caslibs/%s/tables/%s" % (server, caslib, name),
            params=query_params,
        )
        return tbl
