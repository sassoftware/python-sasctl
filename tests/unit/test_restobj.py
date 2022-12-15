#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl import RestObj


def test_restobj():
    o = RestObj(a=1, b=2)

    assert o.a == 1
    assert o["a"] == 1
    assert o.b == 2
    assert o["b"] == 2

    with pytest.raises(AttributeError) as e:
        print(o.missingattribute)
    assert "missingattribute" in str(e.value)

    with pytest.raises(KeyError):
        print(o["c"])

    setattr(o, "c", "attribute")
    assert o.c == "attribute"

    with pytest.raises(KeyError):
        print(o["c"])


def test_repr():
    d = dict(a=1, b=2)

    assert "'a': 1" in repr(RestObj(d))
    assert "'b': 2" in repr(RestObj(d))


def test_str():
    assert str(RestObj(name="test", id=1)) == "test"
    assert str(RestObj(id=1)) == "1"
    assert str({"var": "test"}) in str(RestObj(var="test"))


def test_pickle():
    import pickle

    obj = RestObj(name="test", id=1)

    pickled = pickle.dumps(obj)
    new_obj = pickle.loads(pickled)

    assert obj == new_obj
