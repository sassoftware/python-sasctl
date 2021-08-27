#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from sasctl.services import cas_management as cm


pytestmark = pytest.mark.usefixtures('session')


def test_is_available():
    assert cm.is_available()

    with mock.patch('sasctl._services.cas_management.CASManagement.head') as mocked:
        mocked.return_value.status_code = 404
        assert not cm.is_available()


def test_info():
    info = cm.info()

    assert 'casManagement' == info.serviceId


def test_list_servers():
    servers = cm.list_servers()

    assert isinstance(servers, list)
    assert any(s.name == 'cas-shared-default' for s in servers)


def test_get_server():
    target = 'cas-shared-default'

    server = cm.get_server(target)
    assert target == server.name


def test_list_caslibs():
    caslibs = cm.list_caslibs('cas-shared-default')

    assert isinstance(caslibs, list)
    assert any(l.name == 'Public' for l in caslibs)


def test_get_caslib():
    target = 'Public'
    caslib = cm.get_caslib(target)

    assert target == caslib.name


def test_list_tables():
    tables = cm.list_tables('Public')

    assert isinstance(tables, list)

    assert cm.list_caslibs('fake_caslib') == []


def test_get_table():
    target = 'water_cluster'
    table = cm.get_table(target, 'Samples')

    assert target == table.name.lower()

    assert cm.get_table(target, 'fake_caslib') is None

    assert cm.get_table('fake_table', 'Samples') is None
