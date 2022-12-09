#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import RestObj
from sasctl.services import reports, report_images

pytestmark = pytest.mark.usefixtures("session")


@pytest.mark.incremental
class TestReportImages:
    REPORT_NAME = "Retail Insights"

    def test_get_report(self, request):
        """Verify that the report exists before using it in tests."""
        report = reports.get_report(self.REPORT_NAME)

        assert report.name == self.REPORT_NAME

        # Store the report for subsequent tests
        request.config.cache.set("REPORT", report)

    def test_get_images(self, request):
        report = request.config.cache.get("REPORT", None)
        assert report is not None

        # Cache returns as dict
        report = RestObj(report)

        images = report_images.get_images(report, size=(800, 600))

        assert isinstance(images, list)
        assert len(images) > 0
        assert all(i.startswith(b"<svg ") for i in images)

    def test_get_elements(self, request):
        report = request.config.cache.get("REPORT", None)
        assert report is not None

        # Cache returns as dict
        report = RestObj(report)

        elements = reports.get_visual_elements(report)
        assert isinstance(elements, list)
        assert len(elements) > 0

        graph = [e for e in elements if e.type == "Graph"][0]
        request.config.cache.set("GRAPH", graph)

    def test_get_single_element(self, request):
        report = request.config.cache.get("REPORT", None)
        graph = request.config.cache.get("GRAPH", None)
        assert report is not None
        assert graph is not None

        # Cache returns as dict
        report = RestObj(report)
        graph = RestObj(graph)

        # Get image with default size, specified as string
        images = report_images.get_images(report, size="640x480", elements=graph)
        assert isinstance(images, list)
        assert len(images) == 1

        image = images.pop()
        assert b'width="640"' in image
        assert b'height="480"' in image

        # Get image with default size, specified as tuple
        images = report_images.get_images(report, elements=[graph], size=(640, 480))

        assert isinstance(images, list)
        assert len(images) == 1

        image = images.pop()
        assert b'width="640"' in image
        assert b'height="480"' in image

        # Get image with specific size
        images = report_images.get_images(report, elements=(graph, (800, 600)))

        assert isinstance(images, list)
        assert len(images) == 1

        image = images.pop()
        assert b'width="800"' in image
        assert b'height="600"' in image
