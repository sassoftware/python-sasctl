#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from six.moves import mock

from sasctl.services import ml_pipeline_automation as mpa


def test_create_project_min_args():
    """Create automation project with minimal inputs."""
    from sasctl import current_session

    TABLE = 'table URI'
    TARGET = 'target variable'
    PROJECT = 'project name'

    with mock.patch('sasctl.core.requests.Session.request'):
        current_session('example.com', 'username', 'password')

    with mock.patch(
            'sasctl._services.ml_pipeline_automation.MLPipelineAutomation'
            '.post') as post:
        mpa.create_project(TABLE, TARGET, PROJECT)

    assert post.call_count == 1
    url, data = post.call_args

    # Verify data submitted to SAS
    assert data['json']['dataTableUri'] == TABLE
    assert data['json']['name'] == PROJECT
    assert data['json']['analyticsProjectAttributes'][
               'targetVariable'] == TARGET
