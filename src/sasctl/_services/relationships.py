#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .service import Service


class Relationships(Service):
    """The Relationships API manages a repository of relationships.

    A relationship describes how two resources are connected. A relationship
    contains a subject resource, the related resource, and the relationship
    type.  A relationship type describes the nature of the relationship between
    two resources. The relationship type includes a name, label, and
    description.

    """

    _SERVICE_ROOT = "/relationships"

    (
        list_relationships,
        get_relationship,
        update_relationship,
        delete_relationship,
    ) = Service._crud_funcs("/relationships", "relationship")
