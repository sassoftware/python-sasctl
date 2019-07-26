#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


class AuthenticationError(ValueError):
    """A user could not be authenticated."""

    def __init__(self, username, *args, **kwargs):
        msg = "Authentication failed for user '%s'." % username
        super(AuthenticationError, self).__init__(msg, *args, **kwargs)


