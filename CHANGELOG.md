
Unreleased
----------
 - 
 
v1.2.4 (2019-9-20)
------------------
**Bugfixes**
 - `model_repository.get_model_contents` no longer raises an HTTP 406 error.
 
v1.2.3 (2019-8-23)
------------------
**Changes**
 - `put` request will take an `item` parameter that's used to automatically populate headers for updates.  

**Bugfixes**
 - Convert NaN values to null (None) when calling `microanalytic_score.execute_module_step`.


v1.2.2 (2019-8-21)
------------------
**Bugfixes**
 - `register_model` task should now correctly identify columns when registering a Sci-kit pipeline.
 

v1.2.1 (2019-8-20)
------------------
**Improvements**
 - Added the ability for `register_model` to correctly handle CAS tables containing data step
 score code.
 

v1.2.0 (2019-8-16)
------------------
**Improvements**
- Added `create_model_version` and `list_model_versions` to `model_repository`
- Added an explicit `ValueError` when attempting to register an ASTORE that can't be downloaded.
- Added `start` and `limit` pagination parameters to all default `list_*` service methods.
- Added `create_destination`, `create_cas_destination` and `create_mas_destination` methods for `model_publish` service.

**Changes**
- `Session.add_stderr_logger` default logging level changed to `DEBUG`.

**Bugfixes**
- Fixed an issue where `model_repository` did not find models, projects, or repositories by name once pagination limits were reached. 


v1.1.4 (2019-8-16)
-----------------
**Bugfixes**
 - The `register_model` task now generates dmcas_epscorecode.sas files for ASTORE models.
  

v1.1.3 (2019-8-14)
-----------------
**Bugfixes**
 - Fixed problem causing `register_model` task to include output variables in the input variables list.
 
 
v1.1.2 (2019-8-12)
-----------------
**Improvements**
 - CAS model table automatically reloaded on `publish_model` task.
 
**Bugfixes**
 - Fixed DS2 score code for CAS that was generated when registering a Python model.
 - `PyMAS.score_code(dest='ESP')` corrected to `dest='EP'`
 - Fixed an issue where long user-defined properties prevented model registration.
 
 
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