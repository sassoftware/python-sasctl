
Unreleased
----------
**Changed**
 - Exceptions moved from `sasctl.core` to `sasctl.exceptions`
 - `SWATCASActionError` raised if ASTORE cannot be saved during model registration.
 - Improved handling of MAS calls made via `define_steps()` 


v1.0.0 (2019-07-24)
-------------------
 **Changed**
 - services are now classes instead of modules.
   Imports of services in the format `import sasctl.services.model_management as mm` must be
   changed to `from sasctl.services import model_management as mm`.
 - `host` and `user` parameters of `Session` renamed to `hostname` and `username` to align with SWAT.
 - Only `InsecureRequestWarning` is suppred instead of all `HTTPWarning`
 
 **Improvements**
 - Added `copy_analytic_store` method to `model_repository` service
 - `AuthenticationError` returned instead of `HTTPError` if session authentication fails.


v0.9.7 (2019-07-18)
-------------------
**Improvements**
 - public_model task also defines methods mapped to MAS module steps when publishing to MAS.
 - SSL verification can be disable with `SSLREQCERT` environment variable.
 - CAs to use for validating SSL certificates can also be specified through the `SSLCALISTLOC` environment variable.
 - Added `execute_performance_task`

**Changes**
 - Updated method signature for `create_performance_definition` in Model Manager.

**Bugfixes**
 - register_model task no longer adds `rc` and `msg` variables from MAS to the project variables.


v0.9.6 (2019-07-15)
-------------------
Initial public release.