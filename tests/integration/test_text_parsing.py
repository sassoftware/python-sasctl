#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from sasctl.core import current_session, request_link
from sasctl.services import text_parsing as tp

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
    if current_session().version_info() > 3.5:
        pytest.skip("Text Parsing service was removed in Viya 4.")

    TABLE_NAME = "airline_tweets"
    cas_session.upload(airline_dataset, casout=dict(name=TABLE_NAME, replace=True))
    from sasctl.services import cas_management as cm

    cas_session.table.promote(TABLE_NAME, targetlib="Public")

    input = cm.get_table(TABLE_NAME, "Public")
    job = tp.parse_documents(input, id_column="tweet_id", text_column="text")

    assert_job_succeeds(job)


def test_parsing_inline_docs():
    from sasctl.services import cas_management as cm

    if current_session().version_info() > 3.5:
        pytest.skip("Text Parsing service was removed in Viya 4.")

    caslib = cm.get_caslib("Public")
    input = [
        "Halt! Who goes there?",
        " It is I, Arthur, son of Uther Pendragon, from the castle of "
        "Camelot. King of the Britons, defeater of the Saxons, Sovereign of all England!",
        "Pull the other one!",
        " I am, and this is my trusty servant Patsy. We have ridden the "
        "length and breadth of the land in search of knights who will join me in my court at Camelot. I must speak with your lord and master.",
        "What? Ridden on a horse?",
    ]

    job = tp.parse_documents(input, caslib=caslib, min_doc_count=1)

    assert_job_succeeds(job)


def test_service_removed_error():
    if current_session().version_info() < 4:
        pytest.skip("Text Parsing service was not removed until Viya 4.")

    with pytest.raises(RuntimeError):
        tp.parse_documents("", caslib="Public", min_doc_count=1)
