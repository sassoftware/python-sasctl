#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from six.moves import mock

from sasctl.core import PagedList, RestObj

from .test_pageiterator import paging


def test_len_no_paging():
    items = [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch('sasctl.core.request') as request:
        l = PagedList(obj)
        assert len(l) == 3

        for i, o in enumerate(l):
            assert RestObj(items[i]) == o

    # No request should have been made to retrieve additional data.
    request.assert_not_called()


def test_getitem_no_paging():
    items = [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch('sasctl.core.request') as request:
        l = PagedList(obj)

        for i in range(len(l)):
            item = l[i]
            assert RestObj(items[i]) == item

    # No request should have been made to retrieve additional data.
    request.assert_not_called()


def test_str():
    """Check str formatting of list."""
    source_items = [{'name': 'a'}, {'name': 'b'}, {'name': 'c'},
                    {'name': 'd'}, {'name': 'e'}, {'name': 'f'}]

    start = 2
    limit = 2

    with mock.patch('sasctl.core.request') as req:
        obj = RestObj(items=source_items[:2],
                      count=len(source_items),
                      links=[{'rel': 'next',
                              'href': '/moaritems?start=%d&limit=%d' % (
                                  start, limit)}])

        def side_effect(_, link, **kwargs):
            if 'start=2' in link:
                result = source_items[1:1+limit]
            elif 'start=4' in link:
                result =  source_items[3:3+limit]
            return RestObj(items=result)

        req.side_effect = side_effect

        l = PagedList(obj)

        for i in range(len(source_items)):
            # Force access of each item to ensure it's downloaded
            _ = l[i]

            if i < len(source_items) - 1:
                # Ellipses should indicate unfetched results unless we're
                # at the end of the list
                assert str(l).endswith(', ... ]')
            else:
                assert not str(l).endswith(', ... ]')


def test_getitem_paging(paging):
    """Check that list can be enumerated."""
    obj, items, _ = paging
    l = PagedList(obj)

    # length of list should equal total # of items
    assert len(l) == len(items)

    for i, item in enumerate(l):
        assert item.name == RestObj(items[i]).name


def test_zip_paging(paging):
    """Check that zip() works correctly with the list."""
    obj, items, _ = paging
    l = PagedList(obj)

    # length of list should equal total # of items
    assert len(l) == len(items)

    for target, actual in zip(items, l):
        assert RestObj(target).name == actual.name


def test_slice_paging(paging):
    """Check that [i:j] syntax works correctly with the list."""

    obj, items, _ = paging
    l = PagedList(obj)

    # length of list should equal total # of items
    assert len(l) == len(items)

    # Generate pairs of start:stop indexes that intentionally exceed
    # the size of the array, and include empty sequences.
    starts = range(len(l) + 1)
    stops = range(len(l), -1, -1)

    for start, stop in zip(starts, stops):
        target = items[start:stop]
        actual = l[start:stop]

        for i, item in enumerate(actual):
            assert item.name == RestObj(target[i]).name


def test_copy(paging):
    """Check that [:] syntax works correctly with the list."""
    obj, items, _ = paging
    l = PagedList(obj)

    # length of list should equal total # of items
    assert len(l) == len(items)

    target = items[:]
    actual = l[:]

    assert len(actual) == len(l)

    for i, item in enumerate(actual):
        assert item.name == RestObj(target[i]).name
        print('Verified item %s' % i)
