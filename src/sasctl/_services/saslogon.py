#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .service import Service

# TODO: Class docstring - note admin privs needed.  Reference REST documentation?  Correct sections defined?
# TODO: delete_client() throws 404 if client not found.  Use Service CRUD functions?

class SASLogon(Service):
    """SAS SASLogon service.

    Povides end points for managing client ids and secrets.  This class is somewhat different from the other
    Service classes because many of the operations on the associated SAS SASLogon REST service are related to
    authentication.  In sasctl all authentication is handled in the `Session` class, so only the operations that
    are not related to authentication are implemented here.

    """

    _SERVICE_ROOT = "/SASLogon"

    @classmethod
    def create_client(cls, client_id, client_secret, scopes=None, redirect_uri=None,
                      allow_password=False, allow_client_secret=False, allow_auth_code=False):
        """Register a new client with the SAS Viya environment.

        Parameters
        ----------
        client_id : str
            The ID to be assigned to the client.
        client_secret : str
            The client secret used for authentication.
        scopes : list of string, optional
            Specifies the levels of access that the client will be able to obtain on behalf of users when not using
            client credential authentication.  If `allow_password` or `allow_auth_code` are true, the 'openid' scope
            will also be included.  This is used to assert the identity of the user that the client is acting on
            behalf of.  For clients that only use client credential authentication and therefore do not act on behalf
            of users, the 'uaa.none' scope will automatically be included.
        redirect_uri : str, optional
            The allowed URI pattern for redirects during authorization.  Defaults to 'urn:ietf:wg:oauth:2.0:oob'.
        allow_password : bool, optional
            Whether to allow username & password authentication with this client.  Defaults to false.
        allow_client_secret : bool
            Whether to allow authentication using just the client ID and client secret.  Defaults to false.
        allow_auth_code : bool, optional
            Whether to allow authorization code access using this client.  Defaults to false.

        Returns
        -------
        RestObj

        """

        scopes = set(scopes)

        # Include default scopes depending on allowed grant types
        if allow_password or allow_auth_code:
            scopes.add('openid')
        elif allow_client_secret:
            scopes.add('uaa.none')
        else:
            raise ValueError("At least one authentication method must be allowed.")

        redirect_uri = redirect_uri or 'urn:ietf:wg:oauth:2.0:oob'

        grant_types = set()
        if allow_auth_code:
            grant_types.update(['authorization_code', 'refresh_token'])
        if allow_client_secret:
            grant_types.add('client_credentials')
        if allow_password:
            grant_types.update(['password', 'refresh_token'])

        data = {
            'client_id': client_id,
            'client_secret': client_secret,
            'scope': list(scopes),
            'authorized_grant_types': list(grant_types),
            'redirect_uri': redirect_uri
        }

        # Use access token to define a new client, along with client secret & allowed authorization types (auth code)
        response = cls.post('/oauth/clients', json=data) # , format_='response')

        # if response.status_code == 409:
        #     pass

        # HTTP 201 = data is JSON, convert to RestObj with client info
        # HTTP Error 409: {"error":"invalid_client","error_description":"Client already exists: test_client_2"}
        # HTTP Error 403: {"error":"insufficient_scope","error_description":"Insufficient scope for this resource","scope":"uaa.admin clients.write clients.admin zones.uaa.admin"}

        return response

    @classmethod
    def delete_client(cls, client_id):
        """Remove and existing client.

        Parameters
        ----------
        client_id : str
            The id of the client to delete.

        Returns
        -------
        RestObj
            The deleted client

        """
        return cls.delete(f'/oauth/clients/{client_id}')

    @classmethod
    def get_client(cls, client_id):
        """Retrieve information about a specific client

        Parameters
        ----------
        client_id : str
            The id of the client.

        Returns
        -------
        RestObj

        """
        return cls.get(f'/oauth/clients/{client_id}')

    @classmethod
    def list_clients(cls):
        # TODO: docstring
        # TODO: page list iterator?
        return cls.get('/oauth/clients')

    @classmethod
    def update_client(cls):
        raise NotImplementedError()

    @classmethod
    def update_client_secret(cls):
        raise NotImplementedError()