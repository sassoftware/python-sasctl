#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


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