#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from sasctl.core import current_session, request_link
from sasctl.services import text_categorization as tc

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


def test_from_table():
    from sasctl.services import cas_management as cm

    pytest.xfail("Need to input model URI.  Where to pull from?")

    input = cm.get_table("COMPLAINTS", "Public")
    job = tc.categorize(
        input, id_column="__uniqueid__", text_column="Consumer_complaint_narrative"
    )

    assert_job_succeeds(job)


def test_service_removed_error():
    if current_session().version_info() < 4:
        pytest.skip("Text Categorization service was not removed until Viya 4.")

    with pytest.raises(RuntimeError):
        tc.categorize("", None, id_column="", text_column="")
