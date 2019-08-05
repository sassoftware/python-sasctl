## How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.


### Issues
If you encounter a defect while using this software, please open up an
issue to track the problem.  See [SUPPORT.md](SUPPORT.md) for more information.


### Submiting a Contribution
All contributions are managed through the standard GitHub pull request process.
Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.
 
 All pull requests must include appropriate test cases to verify the changes.  
 See the [Testing](#Testing) section below for more information on how test cases are configured. 
 
Contributions to this project must be accompanied by a signed
[Contributor Agreement](ContributorAgreement.txt).
You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project.


### Testing
Tests are written using the [py.test](https://docs.pytest.org) 
framework, which also supports the standard unittest package.  To enable
integration testing without requiring a running SAS Viya environment, the
[Betamax](https://pypi.org/project/betamax/) package is used to record and
replay network interactions.

In addition, [Tox](https://tox.readthedocs.io) is used to automate common development tasks
such as testing, linting, and building documentation.

All packages required for development and testing are listed in
[tox.ini](tox.ini), but it should be unecessary to install these manually.  Running `tox`
from the project root directory will automatically build virtual environments for all
Python interpreters found on the system and then install the required packages in those environments.

Before a pull request will be accepted:
- contributions must pass existing regression tests located in tests/
- contributions must add unit tests to tests/unit to validate any code changes
- if there's already a test file where your tests would make sense, put them in there
- if it's something new or you feel it needs its own file, create a new file
- contributions should add integration tests to tests/integration when appropriate
- all integration tests that involve network calls should also include the appropriate Betamax cassettes in tests/cassettes

- SASCTL_SERVER_NAME - hostname of the SAS Viya server to be used for testing 
- SASCTL_AUTHINFO - path to .authinfo or a .netrc file containing authentication credentials 
- SASCTL_USER_NAME - the user name to use when authenticating to SAS Viya services
- SASCTL_PASSWORD - the password to use when authenticating to the SAS Viya services
- REQUESTS_CA_BUNDLE - path to CA certificate for the SAS Viya server's SSL certificate.  This is required certificates served by SAS Viya environments
are almost always signed by an internal CA and the Python requests module does not pull CA certificates from the host. 


A collection of py.test fixtures has been defined in [conftest.py](tests/conftest.py) and can be
used to access common resources from test cases.