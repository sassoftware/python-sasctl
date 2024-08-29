#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from sasctl.core import PageIterator, RestObj


@pytest.fixture(
    scope="function", params=[(6, 2, 2), (6, 1, 4), (6, 5, 4), (6, 6, 2), (100, 10, 20)]
)
def paging(request):
    """Create a RestObj designed to page through a collection of items and the
    collection itself.

    Returns
    -------
    RestObj : initial RestObj that can be used to initialize a paging iterator
    List[dict] : List of items being used as the "server-side" source
    MagicMock : Mock of sasctl.request for performing additional validation

    """
    import math
    import re

    num_items, start, limit = request.param

    with mock.patch("sasctl.core.request") as req:
        items = [{"name": str(i)} for i in range(num_items)]

        obj = RestObj(
            items=items[:start],
            count=len(items),
            links=[
                {
                    "rel": "next",
                    "href": "/moaritems?start=%d&limit=%d" % (start, limit),
                }
            ],
        )

        def side_effect(_, link, **kwargs):
            assert "limit=%d" % limit in link
            start = int(re.search(r"(?<=start=)[\d]+", link).group())
            return RestObj(items=items[start : start + limit])

        req.side_effect = side_effect
        yield obj, items[:], req

    # Enough requests should have been made to retrieve all the data.
    # Additional requests may have been made by workers to non-existent pages.
    call_count = (num_items - start) / float(limit)
    assert req.call_count >= math.ceil(call_count)


def test_no_paging_required():
    """If "next" link not present, current items should be included."""

    items = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch("sasctl.core.request") as req:
        # Mock appears to be *sometimes* shared with mock created in `paging` fixture
        # above.  Depending on execution order, this can result in calls to the mock
        # made by other tests to be counted here.  Explicitly reset to prevent this.
        req.reset_mock()
        pager = PageIterator(obj)

        # Returned page of items should preserve item order
        items = next(pager)
        for idx, item in enumerate(items):
            assert item.name == RestObj(items[idx]).name

    # No req should have been made to retrieve additional data.
    try:
        req.assert_not_called()
    except AssertionError as e:
        raise AssertionError(
            f"method_calls={req.mock_calls}  call_args={req.call_args_list}"
        )


def test_paging_required(paging):
    """Requests should be made to retrieve additional pages."""
    obj, items, _ = paging

    pager = PageIterator(obj)
    init_count = pager._start

    for i, page in enumerate(pager):
        for j, item in enumerate(page):
            if i == 0:
                item_idx = j
            else:
                # Account for initial page size not necessarily being same size
                # as additional pages
                item_idx = init_count + (i - 1) * pager._limit + j
            target = RestObj(items[item_idx])
            assert item.name == target.name
