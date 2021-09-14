#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from unittest import mock

import pytest
from sasctl import RestObj
from sasctl.services import job_definitions

# Each test will receive a (recorded) Session instance
pytestmark = pytest.mark.usefixtures('session')


@pytest.mark.incremental
class TestJobDefinitions:
    def test_create_definition(self):

        params = dict(
            name='sasctl_test_job',
            description='Test Job Definition from sasctl',
            type_='Compute',
            code='proc print data=&TABLE;',
            parameters=[
                dict(name='_contextName',
                     defaultValue='SAS Studio compute context',
                     type='character',
                     label='Context Name'),
                dict(name='TABLE',
                     type='character',
                     label='Table Name',
                     required=True)
            ]
        )
        definition = job_definitions.create_definition(**params)

        assert isinstance(definition, RestObj)
        assert definition['name'] == params['name']
        assert definition['description'] == params['description']

    def test_create_definition_bad_param(self):
        params = dict(
            name='sasctl_test_job',
            type_='Compute',
            code='proc print data=&TABLE;',
            parameters=[
                dict(name='TABLE',
                     type='decimal',
                     label='Table Name',
                     required=True)
            ]
        )

        with pytest.raises(ValueError) as e:
            job_definitions.create_definition(**params)

        assert "'DECIMAL'" in str(e.value)
