#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import mock

import pytest
from sasctl import current_session, HTTPError, RestObj
from sasctl.services import saslogon


# Success is RestObj (assuming response is auto-parsed)
# <class 'sasctl.core.RestObj'>(headers={'Date': 'Sun, 18 Dec 2022 18:49:01 GMT', 'Content-Type': 'application/json', 'Transfer-Encoding': 'chunked',
# 'Connection': 'keep-alive', 'Vary': 'Origin, Access-Control-Request-Method, Access-Control-Request-Headers', 'Cache-Control': 'no-cache, no-store, max-age=0, must-revalidate', 'Pragma': 'no-cache', 'Expires': '0', 'Strict-Transport-Security': 'max-age=15724800; includeSubDomains', 'X-XSS-Protection': '1; mode=block', 'X-Frame-Options': 'DENY', 'X-Content-Type-Options': 'nosniff'},
# data={'scope': ['uaa.none'], 'client_id': 'sasctl_client4', 'resource_ids': ['none'], 'authorized_grant_types': ['client_credentials'], 'redirect_uri': ['urn:ietf:wg:oauth:2.0:oob'], 'autoapprove': [], 'authorities': ['uaa.none'], 'lastModified': 1671389341375, 'required_user_groups': []})

# HTTPError raised if duplicate client (if auto-parsed and admin)
# HTTP Error 409: {"error":"invalid_client","error_description":"Client already exists: sasctl_client4"}

# HTTPError raised if insufficient privileges (if auto parsed and not admin)
#  HTTP Error 403: {"error":"insufficient_scope","error_description":"Insufficient scope for this resource","scope":"uaa.admin clients.write clients.admin zones.uaa.admin"}

CLIENT_ID = "SirRobin"
CLIENT_SECRET = "CapitalOfAssyriaIsAssur"


# Set the current session to a dummy session
with mock.patch("sasctl.core.Session._get_authorization_token"):
    current_session("example.com", "username", "password")


@mock.patch("sasctl.core.requests.Session.request")
def test_create_client_password_auth(post):
    """Verify correct HTTP request when creating a client allowing password authentication."""

    # Mock HTTP Response
    post.return_value.status_code = 200
    post.return_value.json.return_value = {
        "scope": ["uaa.none"],
        "client_id": CLIENT_ID,
        "authorized_grant_types": [""],
    }

    result = saslogon.create_client(CLIENT_ID, CLIENT_SECRET, allow_password=True)

    assert post.call_count == 1

    # Verify POST was made to correct URL
    args, kwargs = post.call_args
    assert args[0] == "post"
    assert args[1].endswith("/SASLogon/oauth/clients")

    # Verify request parameters were correctly specified
    data = args[15]
    assert data["client_id"] == CLIENT_ID
    assert data["client_secret"] == CLIENT_SECRET
    assert "password" in data["authorized_grant_types"]
    assert "refresh_token" in data["authorized_grant_types"]
    assert "openid" in data["scope"]

    # Verify response was returned as a RestObj
    assert isinstance(result, RestObj)
    assert result["scope"] == ["uaa.none"]


@mock.patch("sasctl.core.requests.Session.request")
def test_create_client_unauthorized(post):
    """Creating a client without sufficient permissions should raise an HTTPError."""

    # Mock HTTP Response
    post.return_value.status_code = 403
    post.return_value.json.return_value = {
        "error": "insufficient_scope",
        "error_description": "Insufficient scope for this resource",
        "scope": "uaa.admin clients.write clients.admin zones.uaa.admin",
    }

    with pytest.raises(HTTPError):
        saslogon.create_client(CLIENT_ID, CLIENT_SECRET, allow_password=True)

    assert post.call_count == 1

    # Verify POST was made to correct URL
    args, kwargs = post.call_args
    assert args[0] == "post"
    assert args[1].endswith("/SASLogon/oauth/clients")


@mock.patch("sasctl.core.requests.Session.request")
def test_delete_client_success(req):
    """Successful deletion should return deleted client info."""

    CLIENT = {
        "scope": ["uaa.none"],
        "client_id": CLIENT_ID,
        "authorized_grant_types": ["client_credentials"],
    }

    # Successful delete request should response with JSON
    req.return_value.status_code = 200
    req.return_value.json.return_value = CLIENT.copy()

    result = saslogon.delete_client(CLIENT)

    assert req.call_count == 1
    args, _ = req.call_args

    assert args[0] == "delete"
    assert args[1].endswith(f"/SASLogon/oauth/clients/{CLIENT_ID}")

    # Result should be a RestObj with all data from the HTTP response
    assert isinstance(result, RestObj)
    for k in CLIENT:
        assert CLIENT[k] == result[k]


@mock.patch("sasctl.core.requests.Session.request")
def test_delete_client_not_found(req):
    """ValueError should be raised if client id is not found."""

    # Successful delete request should response with JSON
    req.return_value.status_code = 404

    with pytest.raises(ValueError):
        saslogon.delete_client(CLIENT_ID)

    assert req.call_count == 1
    args, _ = req.call_args

    assert args[0] == "delete"
    assert args[1].endswith(f"/SASLogon/oauth/clients/{CLIENT_ID}")


@mock.patch("sasctl.core.requests.Session.request")
def test_list_clients(req):
    """Should return a list of RestObjs containing client details."""
    CLIENTS = [
        {"client_id": "arthur"},
        {"client_id": "lacelot"},
        {"client_id": "patsy"},
    ]

    # Mock HTTP response
    req.return_value.status_code = 200
    req.return_value.json.return_value = {
        "resources": CLIENTS,
        "startIndex": 1,
        "itemsPerPage": 1,
        "totalResults": 25,
    }

    results = saslogon.list_clients(start_index=5, count=10, descending=True)

    assert req.call_count == 1
    args, _ = req.call_args

    assert args[0] == "get"
    assert args[1].endswith("/SASLogon/oauth/clients")
    assert args[2]["startIndex"] == 5
    assert args[2]["sortOrder"] == "descending"
    assert args[2]["count"] == 10

    assert isinstance(results, list)
    assert results == CLIENTS
