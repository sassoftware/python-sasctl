#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service


class Reports(Service):
    """Creates, reads, updates, and deletes reports, report states, and content.

    See Also
    --------
    `REST Documentation <https://developer.sas.com/apis/rest/Visualization/#
    reports>`_

    """

    _SERVICE_ROOT = "reports"

    list_reports, get_report, _, _ = Service._crud_funcs("/reports", "report")

    @classmethod
    def get_visual_elements(cls, report):
        """Get the visual components of a report.

        Returned components may be visualized by rendering with the
        `report_images.get_images` method.

        Parameters
        ----------
        report : str or dict
            The name or id of the report, or a dictionary representation of the
            report.

        Returns
        -------
        List[RestObj]
            List of metadata about each element.

        """
        report = cls.get_report(report)
        elements = cls.request_link(report, "contentVisualElements")

        # Despite being "visual" not all elements can be rendered by
        # report_images service.  Only return renderable elements.
        return [e for e in elements if e.type not in ("Table")]
