#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import RestObj


pytestmark = pytest.mark.usefixtures('session')

PROJECT_NAME = 'Test Project'

def test_list_projects():
    from sasctl.services.projects import list_projects

    projects = list_projects()
    assert all(isinstance(f, RestObj) for f in projects)


def test_get_project():
    from sasctl.services.projects import get_project

    project = get_project('Not a Project')
    assert project is None


def test_create_project():
    from sasctl.services.projects import create_project

    project = create_project(PROJECT_NAME)
    assert isinstance(project, RestObj)
    assert PROJECT_NAME == project.name


def test_delete_project():
    from sasctl.services.projects import delete_project, get_project

    delete_project(PROJECT_NAME)
    project = get_project(PROJECT_NAME)
    assert project is None
