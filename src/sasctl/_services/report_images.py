#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .service import Service
from ..core import RestObj

LOW = 0
MEDIUM = 1
HIGH = 2

_LOD_VALUES = {LOW: "thumbnail", MEDIUM: "normal", HIGH: "entireSection"}


class ReportImages(Service):
    """Delivers SVG images representing elements of a report.

    The images are suitable to the current user, taking row-level-permissions
    and other factors into consideration.

    See Also
    --------
    `REST Documentation <https://developer.sas.com/apis/rest/Visualization/#
    report-images>`_

    """

    _SERVICE_ROOT = "reportImages"

    LOD_LOW = LOW
    """Renders the content with minimal details for creating thumbnails."""

    LOD_MEDIUM = MEDIUM
    """Renders the primary components of the content."""

    LOD_HIGH = HIGH
    """Renders all of the components of content."""

    @classmethod
    def get_images_async(cls, report, section=None, elements=None, size=None, lod=None):
        """Create a job to render images of reports or report sections.

        Report contents are sized and arranged dynamically to accommodate
        browser viewing.  Some or all of these contents may be rendered at any
        time to produce an SVG image of the desired size.  This is done by
        creating a render job which will create the requested images and return
        their download links.

        Parameters
        ----------
        report : str or RestObj
            Report to retrieve images from specified as either a URI or a
            `RestObj` instance as returned from `reports.get_report`.
        section : int, optional
            A specific report section to retrieve, specified by a zero-based
            index.  If not specified, all sections will be returned as separate
            images.
        elements : List[Tuple[str, str]], optional
            Iterable of tuples containing the element name and the size of the
            element when rendered.
        size : str or Tuple[int, int], optional
            Size (in pixels) of the resulting images.  Ignored if `elements` is
            specified.
        lod : int, optional
            Level of detail of the rendered image(s).  Defaults to HIGH.

        Returns
        -------
        RestObj
            The render job

        """

        if isinstance(report, RestObj):
            report = cls.get_link(report, "self")["uri"]

        if isinstance(size, tuple):
            size = "%dx%d" % size

        if size is None:
            size = "640x480"

        lod = lod or HIGH

        if lod not in _LOD_VALUES:
            raise ValueError(
                "LOD value of '%s' is invalid.  Expected one of " "LOW, MEDIUM, HIGH."
            )

        # LOD.HIGH will render the entire section
        # If request is to render individual elements, we must reduce LOD.
        if elements is not None and lod == HIGH:
            lod = MEDIUM

        # Convert integer enum to string value
        lod = _LOD_VALUES[lod]

        formatted_elements = []

        if elements is not None:
            selection = "visualElements"

            # In the case where a single element was passed, wrap it in a list
            if isinstance(elements, (str, dict)):
                elements = [elements]

            # In the case where a single tuple was passed, wrap it in a list
            # Don't do this if a tuple of tuples was passed.
            if isinstance(elements, tuple) and not isinstance(elements[0], tuple):
                elements = [elements]

            for element in elements:
                if isinstance(element, tuple) and len(element) == 2:
                    # Get element & target size if both were provided
                    elem_name, elem_size = element
                else:
                    # Otherwise, assume top-level size value will be used.
                    elem_name = element
                    elem_size = size

                # If element was passed as a RestObj, extract the name
                if hasattr(elem_name, "name"):
                    elem_name = elem_name.name

                # If size was passed as a tuple, format as string
                if isinstance(elem_size, tuple):
                    elem_size = "%dx%d" % elem_size

                formatted_elements.append(dict(name=elem_name, size=elem_size))

        elif section is not None:
            selection = "report"
        else:
            selection = "perSection"

        job = cls.post(
            "/jobs#requestBody",
            json={
                "reportUri": report,
                "layoutType": lod,
                "selectionType": selection,
                "size": size,
                "specificVisualElements": formatted_elements,
                "sectionIndex": section,
            },
        )

        return job

    @classmethod
    def get_images(cls, report, section=None, elements=None, size=None, lod=None):
        """Render images of reports or report sections.

        Report contents are sized and arranged dynamically to accommodate
        browser viewing.  Some or all of these contents may be rendered at any
        time to produce an SVG image of the desired size.  A rendering job will
        automatically be created and the resulting images returned upon job
        completion.

        Parameters
        ----------
        report : str or RestObj
            Report to retrieve images from specified as either a URI or a
            `RestObj` instance as returned from `reports.get_report`.
        section : int, optional
            A specific report section to retrieve, specified by a zero-based
            index.  If not specified, all sections will be returned as separate
            images.
        elements : List[Tuple[str, str]], optional
            Iterable of tuples containing the element name and the size of the
            element when rendered.
        size : str or Tuple[int, int], optional
            Size (in pixels) of the resulting images.  Ignored if `elements` is
            specified.
        lod : `LOD`, optional
            Level of detail of the rendered image(s).  Defaults to LOD.HIGH.

        Returns
        -------
        List[bytes]
            List of binary strings, each representing one SVG image.

        Examples
        --------
        Render a single element of a report

        >>> from sasctl.services import reports
        >>> report = reports.get_report('Parrot Status')
        >>> report_images.get_images(report, elements=('Norwegian Blue', '640x480'))
        [b'<svg>beautiful_plummage</svg>']

        Render each page of the report

        >>> from sasctl.services import reports
        >>> report = reports.get_report('Parrot Status')
        >>> report_images.get_images(report)
        [b'<svg>Customer Complaints</svg>', b'<svg>Deceased</svg>']

        Render the 2nd page of a report

        >>> from sasctl.services import reports
        >>> report = reports.get_report('Parrot Status')
        >>> report_images.get_images(report, section=1)
        [b'<svg>Deceased</svg>']

        """
        job = cls.get_images_async(
            report=report, section=section, elements=elements, size=size, lod=lod
        )

        result = cls._monitor_job(job)

        images = [
            cls.request_link(img, "image", format="content") for img in result.images
        ]

        return images
