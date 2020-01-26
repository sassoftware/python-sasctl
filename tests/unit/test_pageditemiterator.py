#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from six.moves import mock

from sasctl.core import PagedItemIterator, RestObj


def test_no_paging_required():
    """If "next" link not present, current items should be included."""

    items = [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch('sasctl.core.request') as request:
        pager = PagedItemIterator(obj)

        for i, o in enumerate(pager):
            assert RestObj(items[i]) == o

    # No request should have been made to retrieve additional data.
    request.assert_not_called()


def test_is_iterator():
    """PagedItemIterator should be an iterator itself."""
    items = [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch('sasctl.core.request') as request:
        pager = PagedItemIterator(obj)

        for i in range(len(items)):
            o = next(pager)
            assert RestObj(items[i]) == o

    # No request should have been made to retrieve additional data.
    request.assert_not_called()


def test_convert_to_list():
    """Converts correctly to a list."""
    items = [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch('sasctl.core.request') as request:
        pager = PagedItemIterator(obj)

        # Can convert to list
        target = [RestObj(i) for i in items]
        assert list(pager) == target

    # No request should have been made to retrieve additional data.
    request.assert_not_called()


def test_paging(paging):
    """Test that correct paging requests are made."""

    obj, items, request = paging

    pager = PagedItemIterator(obj)

    for i, o in enumerate(pager):
        assert RestObj(items[i]) == o

