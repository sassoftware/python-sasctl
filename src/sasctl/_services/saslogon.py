#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SAS Logon service provides standard OAuth endpoints for client management."""

from ..core import HTTPError
from .service import Service


class SASLogon(Service):
    """The SAS Logon service client management related endpoints.

    Provides functionality for managing client IDs and secrets  This class
    is somewhat different from the other Service classes because many of
    the operations on the associated SA SLogon REST service are related to
    authentication.  In sasctl all authentication is handled in the
    `Session` class, so only the operations that are not related to
    authentication are implemented here.

    The operations provided by this service are only accessible to users
    with administrator permissions.

    """

    _SERVICE_ROOT = "/SASLogon"

    @classmethod
    def create_client(
        cls,
        client_id,
        client_secret,
        scopes=None,
        redirect_uri=None,
        allow_password=False,
        allow_client_secret=False,
        allow_auth_code=False,
    ):
        """Register a new client with the SAS Viya environment.

        Parameters
        ----------
        client_id : str
            The ID to be assigned to the client.
        client_secret : str
            The client secret used for authentication.
        scopes : list of string, optional
            Specifies the levels of access that the client will be able to
            obtain on behalf of users when not using client credential
            authentication.  If `allow_password` or `allow_auth_code` are
            true, the 'openid' scope will also be included.  This is used
            to assert the identity of the user that the client is acting on
            behalf of.  For clients that only use client credential
            authentication and therefore do not act on behalf of users,
            the 'uaa.none' scope will automatically be included.
        redirect_uri : str, optional
            The allowed URI pattern for redirects during authorization.
            Defaults to 'urn:ietf:wg:oauth:2.0:oob'.
        allow_password : bool, optional
            Whether to allow username & password authentication with this
            client.  Defaults to false.
        allow_client_secret : bool
            Whether to allow authentication using just the client ID and
            client secret.  Defaults to false.
        allow_auth_code : bool, optional
            Whether to allow authorization code access using this client.
            Defaults to false.

        Returns
        -------
        RestObj

        """
        scopes = set() if scopes is None else set(scopes)

        # Include default scopes depending on allowed grant types
        if allow_password or allow_auth_code:
            scopes.add("openid")
        elif allow_client_secret:
            scopes.add("uaa.none")
        else:
            raise ValueError("At least one authentication method must be allowed.")

        redirect_uri = redirect_uri or "urn:ietf:wg:oauth:2.0:oob"

        grant_types = set()
        if allow_auth_code:
            grant_types.update(["authorization_code", "refresh_token"])
        if allow_client_secret:
            grant_types.add("client_credentials")
        if allow_password:
            grant_types.update(["password", "refresh_token"])

        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": list(scopes),
            "authorized_grant_types": list(grant_types),
            "redirect_uri": redirect_uri,
        }

        # Use access token to define a new client, along with client secret & allowed
        # authorization types (auth code)
        response = cls.post("/oauth/clients", json=data)

        return response

    @classmethod
    def delete_client(cls, client):
        """Remove and existing client.

        Parameters
        ----------
        client : str or RestObj
            The client ID or a RestObj containing the client details.

        Returns
        -------
        RestObj
            The deleted client

        Raises
        ------
        ValueError
            If `client` is not found.

        """
        id_ = client.get("client_id") if isinstance(client, dict) else str(client)

        try:
            return cls.delete(f"/oauth/clients/{id_}")
        except HTTPError as e:
            if e.code == 404:
                raise ValueError(f"Client with ID '{id_}' not found.") from e
            raise

    @classmethod
    def get_client(cls, client_id):
        """Retrieve information about a specific client

        Parameters
        ----------
        client_id : str
            The id of the client.

        Returns
        -------
        RestObj or None

        """
        return cls.get(f"/oauth/clients/{client_id}")

    @classmethod
    def list_clients(cls, start_index=None, count=None, descending=False):
        """Retrieve a details of multiple clients.

        Parameters
        ----------
        start_index : int, optional
            Index of first client to return.  Defaults to 1.
        count : int, optiona;
            Number of clients to retrieve.  Defaults to 100.
        descending : bool, optional
            Whether to clients should be returned in descending order.

        Returns
        -------
        list of dict
            Each dict contains details for a single client.  If no
            clients were found and empty list is returned.

        """
        params = {}
        if start_index:
            params["startIndex"] = int(start_index)
        if count:
            params["count"] = int(count)
        if descending:
            params["sortOrder"] = "descending"

        results = cls.get("/oauth/clients", params=params)

        if results is None:
            return []

        # Response does not conform to format expected by PagedList (items
        # under an 'items' property and a URL to request additional items).
        # Instead, just return the raw list.
        return results["resources"]

    @classmethod
    def update_client_secret(cls, client, secret):
        """

        Parameters
        ----------
        client : str or RestObj
            The client ID or a RestObj containing the client details.
        secret : str
            The new client secret.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `client` is not found.

        """
        id_ = client.get("client_id") if isinstance(client, dict) else str(client)

        data = {"secret": secret}

        try:
            # Ignoring response ({"status": "ok", "message": "secret updated"})
            _ = cls.put(f"/oauth/clients/{id_}/secret", json=data)
        except HTTPError as e:
            if e.code == 404:
                raise ValueError(f"Client with ID '{id_}' not found.") from e
            raise
