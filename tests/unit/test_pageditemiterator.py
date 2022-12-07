#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

from sasctl.core import PagedItemIterator, RestObj
from .test_pageiterator import paging


def test_no_paging_required():
    """If "next" link not present, current items should be included."""

    items = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch("sasctl.core.request") as request:
        pager = PagedItemIterator(obj)

        for i, o in enumerate(pager):
            assert RestObj(items[i]) == o

    # No request should have been made to retrieve additional data.
    request.assert_not_called()


def test_is_iterator():
    """PagedItemIterator should be an iterator itself."""
    items = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch("sasctl.core.request") as request:
        pager = PagedItemIterator(obj)

        for i in range(len(items)):
            o = next(pager)
            assert RestObj(items[i]) == o

    # No request should have been made to retrieve additional data.
    request.assert_not_called()


def test_convert_to_list():
    """Converts correctly to a list."""
    items = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch("sasctl.core.request") as request:
        pager = PagedItemIterator(obj)

        # Can convert to list
        target = [RestObj(i) for i in items]
        assert list(pager) == target

    # No request should have been made to retrieve additional data.
    request.assert_not_called()


def test_paging(paging):
    """Test that correct paging requests are made."""

    obj, items, _ = paging

    pager = PagedItemIterator(obj)

    for i, o in enumerate(pager):
        assert RestObj(items[i]) == o


def test_paging_inflated_count():
    """Test behavior when server overestimates the number of items available."""
    import re

    start = 10
    limit = 10

    # Only defines 20 items to return
    pages = [
        [{"name": x} for x in list("abcdefghi")],
        [{"name": x} for x in list("klmnopqrs")],
        [{"name": x} for x in list("uv")],
    ]
    actual_num_items = sum(len(page) for page in pages)

    # services (like Files) may overestimate how many items are available.
    # Simulate that behavior
    num_items = 23

    obj = RestObj(
        items=pages[0],
        count=num_items,
        links=[
            {"rel": "next", "href": "/moaritems?start=%d&limit=%d" % (start, limit)}
        ],
    )

    with mock.patch("sasctl.core.request") as req:

        def side_effect(_, link, **kwargs):
            assert "limit=%d" % limit in link
            start = int(re.search(r"(?<=start=)[\d]+", link).group())
            if start == 10:
                return RestObj(items=pages[1])
            elif start == 20:
                return RestObj(items=pages[2])
            else:
                return RestObj(items=[])

        req.side_effect = side_effect

        pager = PagedItemIterator(obj, threads=1)

        # Initially, length is estimated based on how many items the server says it has
        assert len(pager) == num_items

        # Retrieve all of the items
        items = [x for x in pager]

    assert len(items) == actual_num_items
    assert len(pager) == num_items - actual_num_items
