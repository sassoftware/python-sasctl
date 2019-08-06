
Unreleased
----------

 - 

 
v1.1.1 (2019-8-6)
-----------------
**Bugfixes**
- Fixed an issue where usernames were not parsed correctly from .authinfo files, resulting in failed logins. 


v1.1.0 (2019-8-5)
-----------------
 **Improvements**
- Added `update_module` and `delete_module` methods to MAS service.

**Changed**
- Added `replace` parameter to `sasctl.tasks.publish_model` 
- `Session` hostname's can now be specified in HTTP format: 'http://example.com'.

**Bugfixes**
- Renamed `microanalytic_store` service to `microanalytic_score` 


v1.0.1 (2019-07-31)
-------------------
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