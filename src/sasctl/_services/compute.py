#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service


class Compute(Service):
    """The Compute service affords CRUD operations for Compute Contexts.

    A Compute context is analogous to the SAS Application Server from SAS V9.
    """

    _SERVICE_ROOT = '/compute'

    list_contexts, get_context, update_context, delete_context = Service._crud_funcs(
        '/contexts'
    )
