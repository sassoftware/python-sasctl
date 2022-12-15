#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest


def test_request():
    from sasctl.core import request, RestObj

    with mock.patch("sasctl.core.Session") as mock_sess:
        mock_sess.request.return_value.status_code = 200
        mock_sess.request.return_value.json.return_value = dict()

        resp = request("GET", "example.com", session=mock_sess)

    assert mock_sess.request.call_count == 1
    assert isinstance(resp, RestObj)
    assert hasattr(resp, "_headers")


def test_crud_function_doc():
    from sasctl.core import _build_crud_funcs

    for func in _build_crud_funcs("/widgets"):
        assert " widgets " in func.__doc__
        assert "{item}" not in func.__doc__


def test_list_items():
    from sasctl.core import _build_crud_funcs, RestObj

    list_items, _, _, _ = _build_crud_funcs("/items")

    with mock.patch("sasctl.core.request") as request:
        request.return_value = RestObj()

        resp = list_items()

    assert request.call_count == 1
    assert [RestObj()] == resp


def test_get_item_by_dict():
    from sasctl.core import _build_crud_funcs

    _, get_item, _, _ = _build_crud_funcs("/widget")

    # No REST call needed if complete dictionary is passed
    target = {"name": "Test Widget", "id": 12345}
    with mock.patch("sasctl.core.request") as request:
        resp = get_item(target)

    assert target == resp
    assert request.call_count == 0


def test_get_item_by_name():
    from sasctl.core import _build_crud_funcs

    _, get_item, _, _ = _build_crud_funcs("/widget")

    target = {"name": "Test Widget", "id": 12345}
    with mock.patch("sasctl.core.request") as request:
        with mock.patch("sasctl.core.is_uuid") as is_uuid:
            is_uuid.return_value = False
            request.return_value = target
            resp = get_item(target["name"])

    assert target == resp


def test_get_item_by_id():
    from sasctl.core import _build_crud_funcs

    _, get_item, _, _ = _build_crud_funcs("/widget")

    target = {"name": "Test Widget", "id": 12345}
    with mock.patch("sasctl.core.request") as request:
        with mock.patch("sasctl.core.is_uuid") as is_uuid:
            is_uuid.return_value = True
            request.return_value = target
            resp = get_item(12345)

    assert is_uuid.call_count == 1
    assert request.call_count == 1
    assert target == resp


def test_update_item():
    from sasctl.core import _build_crud_funcs, RestObj

    _, _, update_item, _ = _build_crud_funcs("/widget")

    target = RestObj({"name": "Test Widget", "id": 12345})

    with mock.patch("sasctl.core.request") as request:
        request.return_value = target

        # ETag should be required
        with pytest.raises(ValueError):
            resp = update_item(target)

        target._headers = {"etag": "abcd"}
        resp = update_item(target)
    assert request.call_count == 1
    assert ("put", "/widget/12345") == request.call_args[0]
    assert target == resp


def test_put_restobj():
    from sasctl.core import put, RestObj

    url = "/jobDefinitions/definitions/717331fa-f650-4e31-b9e2-6e6d49f66bf9"
    obj = RestObj({"_headers": {"etag": 123, "content-type": "spam"}})

    # Base case
    with mock.patch("sasctl.core.request") as req:
        put(url, obj)

    assert req.called
    args = req.call_args[0]
    kwargs = req.call_args[1]

    assert args == ("put", url)
    assert kwargs["json"] == obj
    assert kwargs["headers"]["If-Match"] == 123
    assert kwargs["headers"]["Content-Type"] == "spam"

    # Should merge with explicit headers
    with mock.patch("sasctl.core.request") as req:
        put(url, obj, headers={"encoding": "spammy"})

    assert req.called
    args = req.call_args[0]
    kwargs = req.call_args[1]

    assert args == ("put", url)
    assert kwargs["json"] == obj
    assert kwargs["headers"]["If-Match"] == 123
    assert kwargs["headers"]["Content-Type"] == "spam"
    assert kwargs["headers"]["encoding"] == "spammy"

    # Should not overwrite explicit headers
    with mock.patch("sasctl.core.request") as req:
        put(url, obj, headers={"Content-Type": "notspam", "encoding": "spammy"})

    assert req.called
    args = req.call_args[0]
    kwargs = req.call_args[1]

    assert args == ("put", url)
    assert kwargs["json"] == obj
    assert kwargs["headers"]["If-Match"] == 123
    assert kwargs["headers"]["Content-Type"] == "notspam"
    assert kwargs["headers"]["encoding"] == "spammy"


def test_request_formats():
    from requests import Response
    import sasctl
    from sasctl.core import request, RestObj

    response = Response()
    response.status_code = 200
    response._content = '{"name": "test"}'.encode("utf-8")

    with mock.patch("sasctl.core.Session") as mock_sess:
        mock_sess.request.return_value = response
        resp = request("GET", "example.com", session=mock_sess, format="response")
        assert mock_sess.request.call_count == 1
        assert isinstance(resp, Response)

        resp = request("GET", "example.com", session=mock_sess, format="json")
        assert isinstance(resp, dict)
        assert resp["name"] == "test"

        resp = request("GET", "example.com", session=mock_sess, format="text")
        assert resp == '{"name": "test"}'

        resp = request("GET", "example.com", session=mock_sess, format="content")
        assert resp == response._content

        resp = request("GET", "example.com", session=mock_sess, format=None)
        assert isinstance(resp, RestObj)
        assert resp.name == "test"


def test_platform_version():
    from sasctl import platform_version

    with mock.patch("sasctl.services.model_repository.info") as mock_info:
        mock_info.return_value = {"build": {"buildVersion": "3.7.231"}}
        version = platform_version()
    assert version == "3.5"

    with mock.patch("sasctl.services.model_repository.info") as mock_info:
        mock_info.return_value = {"build": {"buildVersion": "3.12.77"}}
        version = platform_version()
    assert version == "4.0"


def test_version_info():
    from sasctl.core import VersionInfo

    # Viya 3 should default to Viya 3.5
    assert VersionInfo(major=3) == VersionInfo(major=3, minor=5)

    # Viya 4 should default to Viya 4.0
    assert VersionInfo(major=4) == VersionInfo(major=4, minor=0)

    # Should be able to compare versions, taking into consideration Viya 4 release numbering
    # NOTE: Switch from YYYY.r.u format to YYYY.MM format started with 2022.09 so all YYYY.MM releases
    # should be more recent than all YYYY.r.u formats.
    assert VersionInfo(major=3, minor=4) < VersionInfo(major=3, minor=5)
    assert VersionInfo(major=3, minor=5) < VersionInfo(major=4)
    assert VersionInfo(major=4, release="2021.1.3") < VersionInfo(
        major=4, release="2021.1.4"
    )
    assert VersionInfo(major=4, release="2021.1.4") < VersionInfo(
        major=4, release="2021.2.0"
    )
    assert VersionInfo(major=4, release="2022.2.3") < VersionInfo(
        major=4, release="2022.09"
    )
    assert VersionInfo(major=4, release="2022.09") < VersionInfo(
        major=4, release="2022.10"
    )
    assert VersionInfo(major=4, release="2022.10") < VersionInfo(
        major=4, minor=1, release="2022.01"
    )

    # Should be able to compare with int/float shorthand.
    assert VersionInfo(major=3, minor=4) < 3.5
    assert VersionInfo(major=3, minor=5) == 3.5
    assert VersionInfo(major=4) == 4
    assert VersionInfo(major=4, release="2022.09") == 4
    assert VersionInfo(major=4) >= 4
    assert VersionInfo(major=3) <= 4
