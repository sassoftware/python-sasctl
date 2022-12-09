#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from sasctl.services import cas_management as cm


pytestmark = pytest.mark.usefixtures("session")


def test_is_available():
    assert cm.is_available()

    with mock.patch("sasctl._services.cas_management.CASManagement.head") as mocked:
        mocked.return_value.status_code = 404
        assert not cm.is_available()


def test_info():
    info = cm.info()

    assert "casManagement" == info.serviceId


def test_list_servers():
    servers = cm.list_servers()

    assert isinstance(servers, list)
    assert any(s.name == "cas-shared-default" for s in servers)


def test_get_server():
    target = "cas-shared-default"

    server = cm.get_server(target)
    assert target == server.name


def test_list_caslibs():
    caslibs = cm.list_caslibs("cas-shared-default")

    assert isinstance(caslibs, list)
    assert any(l.name == "Public" for l in caslibs)


def test_get_caslib():
    target = "Public"
    caslib = cm.get_caslib(target)

    assert target == caslib.name


def test_list_tables():
    tables = cm.list_tables("Public")

    assert isinstance(tables, list)

    assert cm.list_caslibs("fake_caslib") == []


def test_get_table():
    target = "water_cluster"
    table = cm.get_table(target, "Samples")

    assert target == table.name.lower()

    assert cm.get_table(target, "fake_caslib") is None

    assert cm.get_table("fake_table", "Samples") is None


def test_list_sessions():
    from sasctl.core import PagedList, RestObj

    sessions = cm.list_sessions()

    # More than one session or just one session
    assert isinstance(sessions, PagedList) or isinstance(sessions, RestObj)


def test_create_session():

    properties = {"authenticationType": "OAuth", "name": "SessionSimulation"}
    sess = cm.create_session(properties)

    assert str(sess["name"]).startswith("SessionSimulation")


def test_delete_session():

    properties = {"authenticationType": "OAuth", "name": "SessionSimulation"}
    sess = cm.create_session(properties)

    cm.delete_session(sess.id)

    query_param = {"filter": "eq(id,'{}')".format(sess.id)}
    res = cm.list_sessions(query_param)

    # Assert that when the created session is deleted, filtered list_sessions only returns an empty list
    assert not res


@pytest.fixture(scope="session")
def sample_table(tmpdir_factory):
    """Create a temporary folder and
    save a CSV file representing the table.
    Return the path to the file.
    """
    path = tmpdir_factory.mktemp("data") / "testtable.csv"
    tbl = "A;B\r\nentry1;entry2\r\nentry3;entry4"
    path.write(tbl)
    return path


def test_upload_file(sample_table):
    properties = {"authenticationType": "OAuth", "name": "SessionSimulation"}
    sess = cm.create_session(properties)

    path = sample_table
    test_tbl = "TEST_TABLE"
    caslib = "Public"
    server = "cas-shared-default"
    frmt = "csv"

    info = {"sessionId": sess.id, "delimiter": ";", "scope": "session"}
    tbl = cm.upload_file(path, test_tbl, caslib, server, True, frmt, detail=info)
    assert tbl.state == "loaded"
    qp = {"sessionId": sess.id}
    r = cm.update_state_table("unloaded", test_tbl, caslib, server, query_params=qp)
    assert r == "unloaded"

    info = {
        "sessionId": sess.id,
        "delimiter": ";",
        "scope": "session",
        "parameter": "wrong",
    }
    with pytest.raises(ValueError):
        cm.upload_file(path, test_tbl, caslib, server, True, frmt, detail=info)

    info = {
        "sessionId": sess.id,
        "delimiter": ";",
        "scope": "session",
        "encoding": "utf-8",
        "password": "pass",
    }
    with pytest.raises(ValueError):
        cm.upload_file(path, test_tbl, caslib, server, True, frmt, detail=info)

    cm.delete_session(sess.id, server)


def test_save_table(sample_table):
    path = sample_table
    table = "TEST_TABLE"
    caslib = "Public"
    server = "cas-shared-default"
    frmt = "csv"
    info = {"delimiter": ";"}
    tbl = cm.upload_file(path, table, caslib, server, True, frmt, detail=info)
    assert tbl.state == "loaded"

    save_tbl = cm.save_table(table, caslib)
    source_tbl_name = save_tbl.tableReference.sourceTableName
    assert source_tbl_name == "%s.sashdat" % (table)

    cm.update_state_table("unloaded", table, caslib, server)


def test_del_table():
    table = "TEST_TABLE"
    caslib = "Public"
    server = "cas-shared-default"
    qp_err = {"sourceTableName": "%s.sashdat" % table}
    with pytest.raises(ValueError):
        cm.del_table(table, qp_err, caslib, server)

    qp = {
        "sourceTableName": "%s.sashdat" % table,
        "quiet": True,
        "removeAcs": True,
    }
    del_tbl = cm.del_table(table, qp, caslib, server)
    assert del_tbl == ""

    get_tbl = cm.get_table(table, caslib, server)
    assert get_tbl is None


def test_promote_table(sample_table):
    properties = {"authenticationType": "OAuth", "name": "SessionSimulation"}
    sess = cm.create_session(properties)

    path = sample_table
    table = "TEST_TABLE"
    caslib = "Public"
    server = "cas-shared-default"
    frmt = "csv"
    info = {"sessionId": sess.id, "delimiter": ";", "scope": "session"}
    tbl = cm.upload_file(path, table, caslib, server, True, frmt, detail=info)
    assert tbl.state == "loaded"

    promote = cm.promote_table(table, sess.id, caslib, server)
    assert promote == "global"

    cm.update_state_table("unloaded", table, caslib, server)
    cm.delete_session(sess.id, server)
