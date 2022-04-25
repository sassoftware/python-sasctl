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


class JobTimeoutError(RuntimeError):
    pass


class ServiceUnavailableError(RuntimeError):
    """A required SAS service is unavailable.

    Raised when correct execution depends on a SAS Viya service that is
    unavailable.  This could be because the necessary SAS components have not
    been licensed or installed, or because the service is temporarily offline.
    """

    def __init__(self, service, *args, **kwargs):
        msg = "The service at '%s' is unavailable." % service._SERVICE_ROOT
        super(ServiceUnavailableError, self).__init__(msg, *args, **kwargs)
