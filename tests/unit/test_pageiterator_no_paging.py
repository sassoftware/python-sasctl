import unittest.mock
from unittest import mock

from sasctl.core import PageIterator, RestObj


def test_no_paging_required():
    """If "next" link not present, current items should be included."""

    items = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    obj = RestObj(items=items, count=len(items))

    import sasctl
    assert not isinstance(sasctl.core.request, unittest.mock.Mock)

    with mock.patch("sasctl.core.request") as req:
        req.reset_mock(side_effect=True)
        try:
            req.assert_not_called()
        except AssertionError:
            raise AssertionError(
                f"Previous calls: method_calls={req.mock_calls}  call_args={req.call_args_list}"
            )

        pager = PageIterator(obj)

        # Returned page of items should preserve item order
        items = next(pager)
        for idx, item in enumerate(items):
            assert item.name == RestObj(items[idx]).name

    # No req should have been made to retrieve additional data.
    try:
        req.assert_not_called()
    except AssertionError:
        raise AssertionError(
            f"method_calls={req.mock_calls}  call_args={req.call_args_list}"
        )
