
sasctl
========

Version |version|

.. contents::
    :local:

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
- six

The following additional packages are recommended for full functionality:

- swat


All required and recommended packages are listed in `requirements.txt` and can be installed easily with::

    pip install -r requirements.txt


Installation
------------

For basic functionality::

    pip install sasctl

For full functionality::

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
    ...    register_model('Sklearn Model', model, 'My Project')


A slightly more low-level way to interact with the environment is to use the service
methods directly::

    >>> from pprint import pprint
    >>> from sasctl import Session, folders

    >>> with Session(host, username, password):
    ...    folders = folders.list_folders()
    ...    pprint(folders)

    {'links': [{'href': '/folders/folders',
                'method': 'GET',
                'rel': 'folders',

    ...  # truncated for clarity

                'rel': 'createSubfolder',
                'type': 'application/vnd.sas.content.folder',
                'uri': '/folders/folders?parentFolderUri=/folders/folders/{parentId}'}],
     'version': 1}


The most basic way to interact with the server is simply to call REST functions
directly, though in general, this is not recommended.::

    >>> from pprint import pprint
    >>> from sasctl import Session, get

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

Registering a model built with SWAT.

.. literalinclude:: ../examples/astore_model.py
   :caption: examples/astore_model.py
   :lines: 7-

Registering a model built with sci-kit learn.

.. literalinclude:: ../examples/sklearn_model.py
   :caption: examples/sklearn_model.py
   :lines: 7-


- publish a model
- score a model

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


The final method for supplying credentials is also simple and straight-forward: environment variables.

**sasctl** recognizes the following authentication-related environment variables:

 - :envvar:`SASCTL_SERVER_NAME`
 - :envvar:`SASCTL_USER_NAME`
 - :envvar:`SASCTL_PASSWORD`




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

Coming soon.


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
The .rst files are then processed by `Sphinx`_ to produce the final documentation.

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
#. Ensure you're environment has the necessary packages:

  :command:`pip install -r test_requirements.txt`

3. Run all unit and integration tests and ensure they pass.  If any tests fail, you should investigate and correct the failure *before* making any changes.
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


Release History
---------------

.. toctree::

   releases

