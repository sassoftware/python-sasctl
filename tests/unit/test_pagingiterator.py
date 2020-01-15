#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from six.moves import mock

from sasctl.core import PagingIterator, RestObj


def test_no_paging_required():
    """If "next" link not present, current items should be included."""
    items = [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch('sasctl.core.request') as request:
        pager = PagingIterator(obj)

        for i, o in enumerate(pager):
            assert RestObj(items[i]) == o

    # No request should have been made to retrieve additional data.
    request.assert_not_called()


def test_convert_to_list():
    """If "next" link not present, current items should be included."""
    items = [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]
    obj = RestObj(items=items, count=len(items))

    with mock.patch('sasctl.core.request') as request:
        pager = PagingIterator(obj)

        # Can convert to list
        target = [RestObj(i) for i in items]
        assert list(pager) == target

    # No request should have been made to retrieve additional data.
    request.assert_not_called()



def test_paging():

    items = [
        {'name': 'a'},
        {'name': 'b'},
        {'name': 'c'},
        {'name': 'd'},
        {'name': 'e'},
        {'name': 'f'}
    ]

    obj = RestObj(items=items[:2],
                  count=len(items),
                  links=[{'rel': 'next', 'href': '/moaritems?start=2&limit=2'}])

    def side_effect(verb, link, **kwargs):
        assert 'limit=2' in link
        if 'start=2' in link:
            return RestObj(items=items[2:4])
        elif 'start=4' in link:
            return RestObj(items=items[4:6])

    # Base case
    with mock.patch('sasctl.core.request') as req:
        req.side_effect = side_effect
        pager = PagingIterator(obj)

        # Can convert to list
        target = [RestObj(i) for i in items]
        assert list(pager) == target

    # No request should have been made to retrieve additional data.
    assert req.call_count == 2

