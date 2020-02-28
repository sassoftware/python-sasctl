#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_fit_statistics():
    from sasctl.utils.metrics import compare
    import numpy as np

    expected = np.random.random(5)
    actual = np.random.random(5)

    stats = compare.fit_statistics(expected, actual)
