#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from sasctl.core import request_link
from sasctl.services import text_parsing as tp

pytestmark = pytest.mark.usefixtures('session')


def assert_job_succeeds(job):
    assert job.state == 'pending'

    while request_link(job, 'state') in ('pending', 'running'):
        time.sleep(1)

    state = request_link(job, 'state')

    if state == 'failed':
        # Refresh to get 'errors' ref
        job = request_link(job, 'self')
        errors = request_link(job, 'errors')
        pytest.fail('Job failed: ' + str(errors))
    assert state == 'completed'


def test_parsing_input_table():
    from sasctl.services import cas_management as cm

    input = cm.get_table('COMPLAINTS', 'Public')
    job = tp.parse_documents(input,
                             id_column='__uniqueid__',
                             text_column='Consumer_complaint_narrative')

    assert_job_succeeds(job)


def test_parsing_inline_docs():
    from sasctl.services import cas_management as cm

    caslib = cm.get_caslib('Public')
    input = [
        'Halt! Who goes there?',
        ' It is I, Arthur, son of Uther Pendragon, from the castle of '
        'Camelot. King of the Britons, defeater of the Saxons, Sovereign of all England!',
        'Pull the other one!',
        ' I am, and this is my trusty servant Patsy. We have ridden the '
        'length and breadth of the land in search of knights who will join me in my court at Camelot. I must speak with your lord and master.',
        'What? Ridden on a horse?'
    ]

    job = tp.parse_documents(input, caslib=caslib, min_doc_count=1)

    assert_job_succeeds(job)
