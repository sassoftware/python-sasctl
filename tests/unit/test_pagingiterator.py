#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
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


@pytest.mark.parametrize('num_items, start, limit', [(6, 2, 2),
                                                     (6, 1, 4),
                                                     (6, 5, 4),
                                                     (6, 6, 2),
                                                     (100, 10, 20)
                                                     ])
def test_paging(num_items, start, limit):
    """Test that correct paging requests are made."""
    import math
    import re

    items = [{'name': str(i)} for i in range(num_items)]

    obj = RestObj(items=items[:start],
                  count=len(items),
                  links=[{'rel': 'next',
                          'href': '/moaritems?start=%d&limit=%d' % (start, limit)}])

    def side_effect(_, link, **kwargs):
        assert 'limit=%d' % limit in link
        start = int(re.search('(?<=start=)[\d]+', link).group())
        return RestObj(items=items[start:start+limit])

    # Base case
    with mock.patch('sasctl.core.request') as req:
        req.side_effect = side_effect
        pager = PagingIterator(obj)

        for i, o in enumerate(pager):
            assert RestObj(items[i]) == o

    # Enough requests should have been made to retrieve all the data.
    call_count = (num_items - start) / float(limit)
    assert req.call_count == math.ceil(call_count)

