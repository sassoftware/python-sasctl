#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2021, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
from sasctl import RestObj
from sasctl.services import compute

# Each test will receive a (recorded) Session instance
pytestmark = pytest.mark.usefixtures('session')


@pytest.mark.incremental
class TestJobDefinitions:

    def test_create_job(self):
        all_contexts = compute.list_contexts()
        assert isinstance(all_contexts, list)
        assert len(all_contexts) > 0

        sess = compute.create_session(all_contexts[0])
        assert isinstance(sess, RestObj)

        code = """
        ods html style=HTMLBlue;
        proc print data=sashelp.class(OBS=5); 
        run; 
        quit;
        ods html close;
        """

        params = dict(
            name='sasctl test job',
            description='Test job execution from sasctl'
        )
        job = compute.create_job(sess, code, **params)
        assert isinstance(job, RestObj)
        assert job.sessionId == sess.id
        assert job.state == 'pending'

    def test_get_listing(self):
        with pytest.raises(ValueError):
            compute.get_listing()

    def test_get_log(self):
        with pytest.raises(ValueError):
            compute.get_log()

    def test_get_results(self):
        with pytest.raises(ValueError):
            compute.get_results()



