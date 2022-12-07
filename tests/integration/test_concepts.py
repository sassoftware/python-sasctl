#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from sasctl.core import request_link
from sasctl.services import concepts as cp


pytestmark = pytest.mark.usefixtures("session")


def assert_job_succeeds(job):
    assert job.state == "pending"

    while request_link(job, "state") in ("pending", "running"):
        time.sleep(1)

    state = request_link(job, "state")

    if state == "failed":
        # Refresh to get 'errors' ref
        job = request_link(job, "self")
        errors = request_link(job, "errors")
        pytest.fail("Job failed: " + str(errors))
    assert state == "completed"


def test_from_table(cas_session, airline_dataset):
    TABLE_NAME = "airline_tweets"
    cas_session.upload(airline_dataset, casout=dict(name=TABLE_NAME, replace=True))
    from sasctl.services import cas_management as cm

    cas_session.table.promote(TABLE_NAME, targetlib="Public")

    input = cm.get_table(TABLE_NAME, "Public")
    job = cp.assign_concepts(input, id_column="tweet_id", text_column="text")

    assert_job_succeeds(job)


def test_from_inline_docs():
    from sasctl.services import cas_management as cm

    caslib = cm.get_caslib("Public")
    input = [
        "Oh yes, the, uh, the Norwegian Blue. What's wrong with it?",
        "I'll tell you what's wrong with it, my lad. He's dead, that's "
        "what's wrong with it!",
        "No, no, he's uh,...he's resting.",
        "Look, matey, I know a dead parrot when I see one, and I'm looking "
        "at one right now.",
        "No no he's not dead, he's, he's resting! Remarkable bird, "
        "the Norwegian Blue, isn't it? Beautiful plumage!",
        "The plumage don't enter into it. It's stone dead.",
    ]

    job = cp.assign_concepts(input, caslib=caslib)

    assert_job_succeeds(job)
