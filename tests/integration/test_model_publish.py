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

    def test_get_publish_destionation(self):
        dest = mp.get_destination('maslocal')

        assert dest.name == 'maslocal'
        assert dest.destinationType == 'microAnalyticService'