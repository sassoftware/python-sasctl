#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl.services import model_publish as mp

# Every test function in the module will automatically receive the session fixture
pytestmark = pytest.mark.usefixtures('session')


@pytest.mark.incremental
class TestModelPublish:
    def test_list_publish_destinations(self):
        destinations = mp.list_destinations()

        assert isinstance(destinations, list)
        assert any(d.name == 'maslocal' for d in destinations)

    def test_get_publish_destination(self):
        dest = mp.get_destination('maslocal')

        assert dest.name == 'maslocal'
        assert dest.destinationType == 'microAnalyticService'

    def test_create_cas_destination(self):
        dest = mp.create_cas_destination('caslocal', 'Public', 'sasctl_models',
                                         description='Test CAS publish destination from sasctl.')

        assert dest.name == 'caslocal'
        assert dest.destinationType == 'cas'
        assert dest.casLibrary == 'Public'
        assert dest.casServerName == 'cas-shared-default'
        assert dest.destinationTable == 'sasctl_models'
        assert dest.description == 'Test CAS publish destination from sasctl.'

    def test_create_mas_destination(self):
        dest = mp.create_mas_destination('maslocal2', 'localhost')

        assert dest.name == 'maslocal2'
        assert dest.destinationType == 'microAnalyticService'
        assert 'description' not in dest