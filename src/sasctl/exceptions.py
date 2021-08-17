#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


class AuthenticationError(ValueError):
    """A user could not be authenticated."""

    def __init__(self, username=None, msg=None, *args, **kwargs):
        if msg is None:
            if username:
                msg = "Authentication failed for user '%s'." % username
            else:
                msg = "Unable to authenticate the user."

        super(AuthenticationError, self).__init__(msg, *args, **kwargs)


class AuthorizationError(RuntimeError):
    """A user lacks permission to perform an action."""
    pass


class JobTimeoutError(RuntimeError):
    pass
