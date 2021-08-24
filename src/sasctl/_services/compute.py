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

    list_servers, _, _, _ = Service._crud_funcs('/servers')

    list_sessions, _, _, _ = Service._crud_funcs('/sessions')

    @classmethod
    def create_job(
        cls,
        session,
        code,
        name=None,
        description=None,
        environment=None,
        variables=None,
        resources=None,
        attributes=None,
    ):
        """

        Parameters
        ----------
        session
        code
        name
        description
        environment
        variables
        resources
        attributes

        Returns
        -------

        """

        # TODO: if code is URI pass in codeUri field.
        code = code.split('\n')

        variables = variables or []
        resources = resources or []
        environment = environment or {}
        attributes = attributes or {}
        data = {
            'version': 3,
            'name': name,
            'description': description,
            'environment': environment,
            'variables': variables,
            'code': code,
            'resources': resources,
            'attributes': attributes,
        }

        return cls.request_link(session, 'execute', json=data)

    @classmethod
    def create_session(cls, context):
        """Create a new session based on an existing Compute Context.

        If a reusable SAS Compute Server is available to handle this session, a new session is created on that SAS
        Compute Server. Otherwise, a new SAS Compute Server is created and the new session is created there.

        Parameters
        ----------
        context : RestObj
            An existing Compute Context as returned by `get_context`.

        Returns
        -------
        RestObj
            Session details
        """
        context = cls.get_context(context)

        return cls.request_link(context, 'createSession')
