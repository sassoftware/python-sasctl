#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl.utils.decorators import deprecated, experimental, ExperimentalWarning


def test_deprecated():
    """Function can be deprecated with @deprecated(version=XX)."""

    @deprecated(version=1.2)
    def old_function(a):
        """Do old stuff.

        Parameters
        ----------
        a : any

        Returns
        -------
        stuff

        """
        return a

    # Calling the function should raise a warning
    with pytest.warns(DeprecationWarning) as warnings:
        result = old_function("spam")

    msg = warnings[0].message
    assert (
        "old_function is deprecated since version 1.2 and may be removed in a future version."
        == str(msg)
    )
    assert ".. deprecated:: 1.2" in old_function.__doc__

    # Return value from function should be unchanged.
    assert result == "spam"


def test_deprecated_with_reason():
    """Function can be deprecated with @deprecated(reason, version=XX)."""

    @deprecated("Use new_function instead.", version=1.3)
    def old_function(a):
        """Do old stuff.

        Parameters
        ----------
        a : any

        Returns
        -------
        stuff

        """
        return a

    # Calling the function should raise a warning
    with pytest.warns(DeprecationWarning) as warnings:
        result = old_function("spam")

    msg = warnings[0].message
    assert (
        "old_function is deprecated since version 1.3 and may be removed in a future version.  Use new_function instead."
        == str(msg)
    )
    assert ".. deprecated:: 1.3\n  Use new_function instead." in old_function.__doc__

    # Return value from function should be unchanged.
    assert result == "spam"


def test_experimental_function():
    @experimental
    def new_function(p):
        """Revive a dead parrot.

        Parameters
        ----------
        p : parrot

        Returns
        -------
        parrot

        """
        return p

    # Calling the function should raise a warning
    with pytest.warns(ExperimentalWarning) as warnings:
        result = new_function("norwegian blue")

    assert ".. warning:: " in new_function.__doc__

    # Return value from function should be unchanged.
    assert result == "norwegian blue"


def test_experimental_class():
    @experimental
    class NewClass:
        pass

    # Calling the function should raise a warning
    with pytest.warns(ExperimentalWarning) as warnings:
        result = NewClass()

    assert ".. warning:: " in NewClass.__doc__
