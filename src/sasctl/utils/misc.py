#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import string

from .decorators import versionadded


def installed_packages():
    """List Python packages installed in the current environment.

    Returns
    -------

    Notes
    -----
    Uses pip freeze functionality so pip module must be present.

    """
    try:
        from pip._internal.operations import freeze
    except ImportError:
        try:
            from pip.operations import freeze
        except ImportError:
            freeze = None

    if freeze is not None:
        return list(freeze.freeze())


@versionadded(version='1.5.1')
def random_string(length):
    """Generates a random alpha-numeric string of a given length.

    Parameters
    ----------
    length : int
        The length of the generate string.

    Returns
    -------
    str

    """

    # random.choices() wasn't added until Python 3.6, so repeatedly call .choice() instead
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))
