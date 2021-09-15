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

    list_servers, get_server, _, stop_server = Service._crud_funcs('/servers')

    _, _, _, delete_session = Service._crud_funcs('/sessions')

    @classmethod
    def list_sessions(cls, server=None):
        """List currently active sessions.

        Parameters
        ----------
        server : str or dict, optional
            The name or id of the server, or a dictionary representation of the server.  If specified, only sessions
            for that server will be returned.  Otherwise, sessions for all running servers are returned.

        Returns
        -------
        list of RestObj

        Raises
        ------
        ValueError
            If `server` is specified but not found.

        """
        if server is not None:
            server_obj = cls.get_server(server)
            if server_obj is None:
                raise ValueError("Unable to find server '%s'." % server)
            uri = '/servers/%s/sessions' % server_obj['id']
        else:
            uri = '/sessions'

        results = cls.get(uri)
        if isinstance(results, list):
            return results

        return [results]

    @classmethod
    def get_server_state(cls, server):
        """

        Parameters
        ----------
        server : str or dict
            The name or id of the server, or a dictionary representation of the server.


        Returns
        -------
        str
            {'running', 'stopped'}

        """
        server_obj = cls.get_server(server)
        uri = '/servers/%s/state' % server_obj['id']

        return cls.get(uri)

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
        session : str or dict
        code : str
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

        uri = '/sessions/%s/jobs' % session['id']

        # NOTE: variables must be None not []
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

        return cls.post(uri, json=data)
        # return cls.request_link(session, 'execute', json=data)

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

    @classmethod
    def get_listing(cls, job=None, session=None):
        """Retrieve the SAS listing (program output) for a job or an entire session.

        Parameters
        ----------
        job : dict, optional
            Dictionary representation of an existing job.
        session : dict, optional
            Dictionary representation of an active session.

        Returns
        -------
        list of RestObj

        Raises
        ------
        ValueError
            If neither `job` nor `session` are provided.

        """
        if job is None and session is None:
            raise ValueError("Either a job or a session must be specified.")

        if job:
            uri = '/sessions/%s/jobs/%s/listing' % (job['sessionId'], job['id'])
        else:
            uri = '/sessions/%s/listing' % session['id']

        return cls.get(uri)

    @classmethod
    def get_log(cls, job=None, session=None):
        """Retrieve the SAS log for a job or an entire session.

        Parameters
        ----------
        job : dict, optional
            Dictionary representation of an existing job.
        session : dict, optional
            Dictionary representation of an active session.

        Returns
        -------
        list of RestObj

        Raises
        ------
        ValueError
            If neither `job` nor `session` are provided.

        """
        if job is None and session is None:
            raise ValueError("Either a job or a session must be specified.")

        if job:
            uri = '/sessions/%s/jobs/%s/log' % (job['sessionId'], job['id'])
        else:
            uri = '/sessions/%s/log' % session['id']

        return cls.get(uri)

    @classmethod
    def get_results(cls, job=None, session=None):
        """Retrieve the output files for a job or an entire session.

        Parameters
        ----------
        job : dict, optional
            Dictionary representation of an existing job.
        session : dict, optional
            Dictionary representation of an active session.

        Returns
        -------
        list of RestObj

        Raises
        ------
        ValueError
            If neither `job` nor `session` are provided.

        """
        if job is None and session is None:
            raise ValueError("Either a job or a session must be specified.")

        if job:
            uri = '/sessions/%s/jobs/%s/results' % (job['sessionId'], job['id'])
        else:
            uri = '/sessions/%s/results' % session['id']

        return cls.get(uri)