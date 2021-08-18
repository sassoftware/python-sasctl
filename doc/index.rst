
sasctl
========

Version |version|

.. toctree::
    :maxdepth: 3
    :hidden:

    index

Introduction
------------

**sasctl** enables easy integration with the SAS Viya platform.

It can be used directly as a python module::

    >>> sasctl.folders.list_folders()

Or as a command line utility::

    $ sasctl folders list


Prerequisites
-------------

**sasctl** requires the following Python packages be installed.
If not already present, these packages will be downloaded and install automatically.

- requests
- pyyaml

The following additional packages are recommended for full functionality:

- swat
- kerberos


Installation
------------

For basic functionality::

    pip install sasctl


Functionality that depends on additional packages can be installed using the following::

    pip install sasctl[swat]
    pip install sasctl[kerberos]
    pip install sasctl[all]


Quickstart
----------

As a Module
++++++++++++

Once the **sasctl** package has been installed and you have a SAS Viya server to
connect to, the first step is to establish a session::

    >>> from sasctl import Session

    >>> s = Session(host, username, password)

Once a session has been created, all commands will target that environment by default.
The easiest way to use **sasctl** is often to use a pre-defined task, which will
handle all necessary communication with the SAS Viya server::

    >>> from sasctl import Session, register_model
    >>> from sklearn import linear_model as lm

    >>> with Session('example.com', authinfo=<authinfo file>):
    ...    model = lm.LogisticRegression()
    ...    register_model(model, 'Sklearn Model', 'My Project')


A slightly more low-level way to interact with the environment is to use the service
methods directly::

    >>> from sasctl import Session
    >>> from sasctl.services import folders

    >>> with Session(host, username, password):
    ...    for f in folders.list_folders():
    ...        print(f)

    Public
    Projects
    ESP Projects
    Risk Environments

    ...  # truncated for clarity

    My Folder
    My History
    My Favorites
    SAS Environment Manager

The most basic way to interact with the server is simply to call REST functions
directly, though in general, this is not recommended.::

    >>> from pprint import pprint
    >>> from sasctl import get, Session

    >>> with Session(host, username, password):
    ...    folders = get('/folders')
    ...    pprint(folders)

    {'links': [{'href': '/folders/folders',
                'method': 'GET',
                'rel': 'folders',

    ...  # truncated for clarity

                'rel': 'createSubfolder',
                'type': 'application/vnd.sas.content.folder',
                'uri': '/folders/folders?parentFolderUri=/folders/folders/{parentId}'}],
     'version': 1}


As a Command Line Utility
+++++++++++++++++++++++++

When using **sasctl** as a command line utility you must pass authentication credentials using environment variables.
See the :ref:`authentication` section for details

.. todo:: Add info for SERVER_NAME env var


Once these environment variables have been set you can simply run **sasctl** and ask for help::

    $ sasctl -h

    usage: sasctl [-h] [-k] [-v] [--version]
                  {folders,performanceTasks,models,repositories,projects} ...

    sasctl interacts with a SAS Viya environment.

    optional arguments:
      -h, --help            show this help message and exit
      -k, --insecure        Skip SSL verification
      -v, --verbose
      --version             show program's version number and exit

    service:
      {folders,performanceTasks,models,repositories,projects}

This also works on individual commands::

    $ sasctl folders -h

    usage: sasctl folders [-h] {create,get,list,update,delete} ...

    optional arguments:
      -h, --help            show this help message and exit

    command:
      {create,get,list,update,delete}
        create
        get                 Returns a folders instance.
        list                List all folders available in the environment.
        update              Updates a folders instance.
        delete              Deletes a folders instance.


Common Uses
++++++++++++

Register a SAS model
~~~~~~~~~~~~~~~~~~~~~~~

  ::

    import swat
    from sasctl import register_model, Session

    with swat.CAS('hostname', 5570, 'username', 'password') as cas:
        astore = cas.CASTable('model_astore_table')

        Session('hostname', 'username', 'password'):

        model = register_model(astore, 'Model Name', 'Project Name')


Register a scikit-learn model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ::

    from sklearn.linear_model import LogisticRegression
    from sasctl import register_model, Session

    model = LogisticRegression()
    model.fit(X, y)

    # Establish a session with Viya
    with Session('hostname', 'username', 'password'):
        register_model(model, 'Model Name', 'Project Name', input=X)


Publish a model
~~~~~~~~~~~~~~~

  ::

    from sasctl import publish_model, Session
    from sasctl.services import model_repository as mr

    model = mr.get_model('Model Name')
    publish_model(model, 'Destination Name')


Execute a model in MAS
~~~~~~~~~~~~~~~~~~~~~~

  ::

    from sasctl import Session
    from sasctl.services import microanalytic_score as mas

    module = mas.get_module('Module Name')
    module = mas.define_steps(module)
    module.predict(model_inputs)



See the :file:`examples/` directory in the repository for more complete examples.




User Guide
----------

.. _authentication:

Authentication
++++++++++++++

There are a variety of ways to provide authentication credentials when creating a :class:`.Session`.  They are presented
here in the order of precedence in which they are recognized.  The simplest method
is to just provide the information directly::

    >>> s = Session(hostname, username, password)

Although this is often the easiest method when getting started, it is not the most secure.  If your program will be used
interactively, consider using the builtin :mod:`getpass` module to avoid hard-coded user names and passwords.

Because **sasctl** augments analytic modeling tasks, it may frequently be used in conjuction with the :mod:`swat`
module.  When this is the case, another easy way to create a :class:`.Session` is to simply reuse the existing CAS
connection from swat::

    >>> cas_session = swat.CAS(hostname, username, password)
    ...
    >>> s = Session(cas_session)

Note: this method will only work when the SWAT connection to CAS is using REST.

A related option is to piggy-back on the `.authinfo` file used by :mod:`swat` or a `.netrc` file::

    >>> s = Session(hostname, authinfo=file_path)

Note: this method will not work with SAS-encoded passwords that may be contained in a :file:`.authinfo` file.

If the SAS Viya server is configured for Kerberos and a TGT is already present on the client, then a session
can be instantiated using simply the hostname:

    >>> s = Session(hostname)

If a username and password are not provided, and the SAS Viya server has **not** been configured for Kerberos then
**sasctl** will attempt to connect using OAuth2 authorization codes.  In this situation, you may be prompted to open
a URL in your browser, retrieve an authorization code, and then enter it before sasctl can connect.

The final method for supplying credentials is also simple and straight-forward: environment variables.

**sasctl** recognizes the following authentication-related environment variables:

 - :envvar:`SASCTL_SERVER_NAME`
 - :envvar:`SASCTL_USER_NAME`
 - :envvar:`SASCTL_PASSWORD`
 - :envvar:`SASCTL_CLIENT_ID`
 - :envvar:`SASCTL_CLIENT_SECRET`



SSL Certificates
++++++++++++++++

By default, **sasctl** will use HTTPS connections to communicating with the server and will validate the certificate
presented by the server against the client's trusted :abbr:`CA (Certificate Authority)` list.

While this behavior should be sufficient for most use cases, there may be times where it is necessary to trust a certificate
that has not been signed by a trusted :abbr:`CA (Certificate Authority)`.  The following environment variables can be used
to accomplish this behavior:

 - :envvar:`CAS_CLIENT_SSL_CA_LIST`
 - :envvar:`REQUESTS_CA_BUNDLE`

In addition, it is possible to disable SSL ceritificate validation entirely, although this should be used with caution.
When instantiating a :class:`.Session` instance you can set the `verify_ssl` parameter to False::

   >>> s = Session(hostname, username, verify_ssl=False)

If you're using **sasctl** from the command line, or want to disable SSL validation for all sessions, you can use the following :envvar:`SSLREQCERT` environment variable.


Logging
+++++++

All logging is handled through the built-in :mod:`logging` module with standard module-level loggers.  The one exception
to this is :class:`.Session` request/response logging.  Sessions contain a :attr:`~sasctl.core.Session.message_log` which is exclusively used
to record requests and responses made through the session.  Message recording can be configured on a per-session basis
by updating this logger, or the ``sasctl.core.session`` logger can be configured to control all message recording by all sessions.



HATEOAS
+++++++

Many of the SAS microservices follow the `HATEOAS`_ paradigm and the standard is for services to return a links
collection containing valid operations.  Most **sasctl** operations return one or more instances of :class:`~sasctl.core.RestObj`.  Any
links related to that object are accessible via the ['links'] key.  Each link is represented as a dictionary containing metadata::

    {'method': 'POST', 'rel':
     'createFolder',
      'href': '/folders/folders',
      'uri': '/folders/folders',
      'type': 'application/vnd.sas.content.folder'}

However, instead of having to parse this collection to find a link, **sasctl** includes some functions to make this easy:  :func:`~sasctl.core.get_link` and :func:`~sasctl.core.request_link`.

Given an object and a link name (`rel`) :func:`~sasctl.core.get_link` will return the metadata for that link.  Similarly,
:func:`~sasctl.core.request_link` will make the request to the link and return the response object.

.. _`HATEOAS`: https://restfulapi.net/hateoas/

API Reference
-------------
.. toctree::
   :maxdepth: 2

   api/sasctl


Environment Variables
+++++++++++++++++++++

.. envvar:: CAS_CLIENT_SSL_CA_LIST

Client-side path to a certificate file containing :abbr:`CA (Certificate Authority)` certificates to be trusted.  Used by the :mod:`swat` module.  This
will take precedence over :envvar:`SSLCALISTLOC` and :envvar:`REQUESTS_CA_BUNDLE`.

.. envvar:: SSLCALISTLOC

Client-side path to a certificate file containing :abbr:`CA (Certificate Authority)` certificates to be trusted.  Used by the :mod:`swat` module.  This
will take precedence over :envvar:`REQUESTS_CA_BUNDLE`.

.. envvar:: REQUESTS_CA_BUNDLE

Client-side path to a certificate file containing :abbr:`CA (Certificate Authority)` certificates to be trusted.  Used by the :mod:`requests` module.

.. envvar:: SSLREQCERT

Disables validation of SSL certificates when set to `no` or `false`

.. envvar:: SASCTL_SERVER_NAME

Hostname of the SAS Viya server to connect to.  Required for CLI usage.

.. envvar:: SASCTL_USER_NAME

The name of the user that will be used when creating the :class:`.Session` instance.

.. envvar:: SASCTL_PASSWORD

Password for authentication to the SAS Viya server.

.. envvar:: SASCTL_CLIENT_ID

OAuth2 client ID used during authorization.

.. envvar:: SASCTL_CLIENT_SECRET

OAuth2 client secret used during authorization.




Contributor Guide
-----------------

Ways to contribute
 - :ref:`improving-the-docs`
 - :ref:`contributing-code`
 - Feature requests
 - Review
 - Triaging issues
 - Reviewing pull requests


.. todo:: feature requests / bug reports
.. todo:: review pull requests
.. todo:: triage issues

.. _improving-the-docs:

Improving the documentation
+++++++++++++++++++++++++++

Accurate, clear, and concise documentation that is also easy to read is critical for the success of any project.  This
makes improving or expanding the **sasctl** documentation one of the most valuable ways to contribute.  In addition,
simple code examples that demonstrate how to use various features are always welcome.  These may be added into the
:file:`examples/` directory or placed inline in the appropriate documentation.


All documentation is contained in the :file:`doc/` directory of the source repository and is written in `reStructuredText`_.
The .rst files are then processed by `Sphinx`_ to produce the final documentation.  See the :ref:`tox_commands` section for
details on how to build the final documentation.

.. _`reStructuredText`: http://docutils.sourceforge.net/rst.html
.. _`Sphinx`: http://www.sphinx-doc.org/en/master/


.. _contributing-code:

Contributing new code
+++++++++++++++++++++

All code contributions are managed through the standard GitHub pull request process.
Consult `GitHub Help`_ for more information on using pull requests.
In addition, all pull requests must include appropriate test cases to verify the changes. See the :ref:`testing` section
for more information on how test cases are configured.

.. _`GitHub Help`: https://help.github.com/articles/about-pull-requests/

Contributions to this project must also be accompanied by a signed :download:`Contributor Agreement <../ContributorAgreement.txt>`. You (or your employer) retain
the copyright to your contribution, this simply gives us permission to use and redistribute your contributions as
part of the project.

1. Fork the repository
#. Run all unit and integration tests and ensure they pass.  This can be easily accomplished by running the following:

  :command:`tox`

  See :ref:`tox_commands` for more information on using Tox.

3. If any tests fail, you should investigate and correct the failure *before* making any changes.
#. Make your code changes
#. Include new tests that validate your changes
#. Rerun all unit and integration tests and ensure they pass.
#. Submit a GitHub pull request


All code submissions must meet the following requirements before the pull request will be accepted:
 - Contain appropriate unit/integration tests to validate the changes
 - Document all public members with `numpydoc`_-style docstrings
 - Adherence to the :pep:`8` style guide

.. _`numpydoc`: https://numpydoc.readthedocs.io/en/latest/format.html



.. _testing:

Testing
+++++++

Automated testing is used to ensure that existing functionality and new features continue to work on all supported
platforms.  To accomplish this, the :file:`tests/` folder contains a collection of unit tests and integration tests where
"unit" tests are generally target a single function and "integration" tests target a series of functions or classes
working together.

Test execution is handled by the  :doc:`py.test <pytest:index>` module which supports tests written using the builtin :mod:`unittest`
framework, but also adds some powerful features like test fixtures.  It is recommended that you review
:file:`tests/conftest.py` and the existing test cases to understand what features are currently available for testing.

To isolate individual methods for testing, the unit test cases make extensive use of mocking via the
builtin :mod:`unittest.mock` module.

Most of the integration tests execute end-to-end functionality that would normally depend on a running SAS Viya
environment.  However, it can be difficult to reliably test against dynamic environments and not all developers may have
access to an environment that is suitable for development.  Therefore, most of the integration tests rely on the
:mod:`betamax` module to record and replay HTTP requests and responses.  These recordings are scrubbed of any sensitive
information and stored in :file:`tests/cassettes/`.  This allows tests to be rerun repeatedly once the test has been recorded
once.

And finally, the :doc:`tox <tox:index>` module is used to ensure that **sasctl** will install and work correctly on all supported
Python versions.


.. _tox_commands:

Useful Tox Commands
+++++++++++++++++++
:mod:`tox`  is used to automate common development tasks such as testing, linting, and building documentation.
Running :program:`tox` from the project root directory will automatically build virtual environments for all Python interpreters
found on the system and then install the required packages necessary to perform a given task.  The simplest way to run Tox is:

.. code::

   $ tox

This will run the :mod:`flake8` linter followed by :mod:`pytest` to test the code against all Python runtimes
found on the machine.  One of the great features of Tox is the ability to run specific tasks by specifying the environment to run.
A few useful environments are listed below, where **XX** indicates a Python version present in your development environment,
such as '27' or '36'.

#.
   .. code::

      $ tox -e pyXX-flake8

   Runs the flake8 linter against all **sasctl** source code.

#.
   .. code::

      $ tox -e pyXX-flake8 src/sasctl/tasks.py

   Runs the flake8 linter against a specific file.

#.
   .. code::

      $ tox -e pyXX-tests

   Runs all tests using the specified Python interpreter.

#.
   .. code::

      $ tox -e pyXX-doc

   Builds the documentation.

#.
   .. code::

      $ tox -e pyXX-tests -- python

   Starts a Python REPL in an environment with **sasctl** already installed.

For additional information on configuring and using Tox, see the official :doc:`documentation <tox:index>` or Sean Hammond's excellent `tutorial`_.

.. _`tutorial`: https://seanh.cc/post/tox-tutorial/

