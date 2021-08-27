#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from sasctl.services import data_sources as ds

pytestmark = pytest.mark.usefixtures('session')


def test_is_available():
    assert ds.is_available()

    with mock.patch('sasctl._services.data_sources.DataSources.head') as mocked:
        mocked.return_value.status_code = 404
        assert not ds.is_available()


def test_info():
    info = ds.info()

    assert 'dataSources' == info.serviceId


def test_list_providers():
    providers = ds.list_providers()

    assert isinstance(providers, list)
    assert 'cas' in [p.id for p in providers]


def test_list_cas_sources():
    sources = ds.list_sources('cas')

    assert isinstance(sources, list)
    assert any(s.name == 'cas-shared-default' for s in sources)


def test_get_cas_source():
    target = 'cas-shared-default'
    source = ds.get_source('cas', target)

    assert target == source.name


def test_list_caslibs():
    cas_server = 'cas-shared-default'

    # Get caslibs using a Source name
    caslibs = ds.list_caslibs(cas_server)

    assert isinstance(caslibs, list)
    assert any(c.name == 'Public' for c in caslibs)

    # Get caslibs using a Source instance
    source = ds.get_source('cas', cas_server)
    caslibs_2 = ds.list_caslibs(source)

    assert isinstance(caslibs_2, list)
    # Force download before comparing
    assert caslibs[:] == caslibs_2[:]


def test_get_caslib():
    target = 'Public'
    caslib = ds.get_caslib(target)

    assert target == caslib.name


def test_list_tables():
    tables = ds.list_tables('Samples')

    assert isinstance(tables, list)
    assert any(t.name == 'WATER_CLUSTER' for t in tables)


def test_get_table():
    target = 'WATER_CLUSTER'
    table = ds.get_table(target, 'Samples')

    assert target == table.name

    table = ds.get_table(target.lower(), 'Samples')

    # Filter is case sensitive
    assert table is None
