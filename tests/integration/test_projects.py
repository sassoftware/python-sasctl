#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import current_session, RestObj
from sasctl.services import projects as proj


pytestmark = pytest.mark.usefixtures("session")

PROJECT_NAME = "Test Project"


def test_list_projects():
    projects = proj.list_projects()
    assert all(isinstance(f, RestObj) for f in projects)


def test_get_project():
    project = proj.get_project("Not a Project")
    assert project is None


def test_create_project():
    if current_session().version_info() >= 4:
        pytest.skip("Projects service was removed from Viya 4.")

    project = proj.create_project(PROJECT_NAME)
    assert isinstance(project, RestObj)
    assert PROJECT_NAME == project.name


def test_delete_project():
    proj.delete_project(PROJECT_NAME)
    project = proj.get_project(PROJECT_NAME)
    assert project is None
