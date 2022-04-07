Unreleased
----------
 -

v1.6.4 (2022-04-07)
-------------------
**Bugfixes**
 - Fixed an issue where `folders.create_folder()` would attempt to use root folder as parent if desired parent
   folder wasn't found.  Now correctly handles parent folders and raises an error if folder not found.

v1.6.3 (2021-09-23)
-------------------
**Bugfixes**
 - Fix an issue where `pzmm.ZipModel.zipFiles()` threw an error on Python 3.6.1 and earlier.
 
v1.6.2 (2021-09-09)
-------------------
**Bugfixes**
 - Fixed an issue with `register_model()` where random forest, gradient boosting, and SVM regression models with 
 nominal inputs where incorrectly treated as classification models. 
 
v1.6.1 (2021-09-01)
-------------------
**Improvements**
 - `model_repository.add_model_content()` will now overwrite existing files instead of failing.
 
**Bugfixes**
 - `PagedList.__repr__()` no longer appears to be an empty list. 

v1.6.0 (2021-06-29)
-------------------
**Improvements**
 - `Session` now supports authorization using OAuth2 tokens.  Use the `token=` parameter in the constructor when 
 an existing access token token is known.  Alternatively, omitting the `username=` and `password=` parameters
 will now prompt the user for an auth code.
 
**Changes**
 - `current_session` now stores & returns the *most recently created* session, not the first created session.  This
 was done to alleviate quirks where an old, expired session is implicitly used instead of a newly-created session.
 - Removed deprecated `raw=` parameter from `sasctl.core.request()`.
 - Dropped support for Python 2.
 
v1.5.9 (2021-06-09)
-------------------
**Bugfixes**
 - Fixed an issue that caused score code generation by `pzmm` module to fail with Viya 3.5.
 
v1.5.8 (2021-05-18)
-------------------
**Bugfixes**
 - SSL warnings no longer repeatedly raised when `verify_ssl=False` but `CAS_CLIENT_SSL_CA_LIST` is specified.
 - `model_repository.delete_model_contents()` no longer fails when only one file is found.
 
**Improvements**
 - All `delete_*()` service methods return `None` instead of empty string.
 - All `get_*()` service methods issue a warning if multiple items are found when retrieving by name.
 
v1.5.7 (2021-05-04)
-------------------
**Bugfixes**
 - Fixed an import issue that could cause an error while using the `pzmm` submodule.

v1.5.6 (2021-04-30)
-------------------
**Improvements**
 - `PagedList` handles situations where the server over-estimates the number of items available for paging.
 - The version of SAS Viya on the server can now be determined using `sasctl.platform_version()`.
 
**Bugfixes**
 - Reworked the `model_repository.get_repository()` to prevent HTTP 403 errors that could occur with some Viya environments.
 
v1.5.5 (2021-03-26)
-------------------
**Bugfixes***
 - Fixed an issue with JSON parsing that caused the `publish_model` task to fail with Viya 4.0.
 
v1.5.4 (2020-10-29)
------------------
**Improvements**
 - Added the `as_swat` method to the `Session` object, allowing connection to CAS through SWAT without an additional authentication step.
 
**Changes**
 - Integrated PZMM into `Session` calls and removed redundant function calls in PZMM.
 - ROC and Lift statistic JSON files created by PZMM are now generated through CAS actionset calls.
 - Updated the PZMM example notebook, `FleetMaintenance.ipynb`, to include integration of PZMM with sasctl functions.
 
**Bugfixes**
 - Reworked the `model_repository.get_repository()` to prevent HTTP 403 errors that could occur with some Viya environments.
 
v1.5.3 (2020-06-25)
------------------
**Bugfixes**
 - Added PZMM fitstat JSON file to manifest.
 
v1.5.2 (2020-06-22)
-------------------
**Improvements**
  - PZMM module moved from a stand-alone [repository](https://github.com/sassoftware/open-model-manager-resources/tree/master/addons/picklezip-mm) to a sasctl submodule.
  - Introduced deprecation warnings for Python 2 users.
 
v1.5.1 (2020-4-9)
----------------
**Bugfixes**
 - Fixed PyMAS utilities to correctly work functions not bound to pickled objects.
 - Model target variables should no longer appear as an input variable when registering ASTORE models. 
 
v1.5 (2020-2-23)
----------------
**Improvements**
 - Registered Python models will now include both `predict` and `predict_proba` methods. 
 - Added a new Relationships service for managing links between objects.
 - Added a new Reports service for retrieving SAS Visual Analytics reports.
 - Added a new Report_Images service for rendering content from reports. 
 - Additional metadata fields are set when registering an ASTORE model.
 - Collections of items should now return an instance of `PagedList` for lazy loading of results.
 - Module steps can now be called using `module.step(df)` where `df` is the row of a DataFrame or Numpy array.
 - `register_model` sets additional project properties when registering an ASTORE model.

**Changes**
 - Replaced the `raw` parameter of the `request` methods with a `format` parameter, allowing more control over the
   returned value.
 - The `get_file_content` method of the Files service now returns the actual content instead of the file metadata.
 - JSON output when using `sasctl` from the command line is now formatted correctly.
 
**Bugfixes**
 - `model_publish.delete_destination` now works correctly.
 
v1.4.6 (2020-1-24)
------------------
**Bugfixes**
 - Fixed an issue where the `REQUESTS_CA_BUNDLE` environment variable was taking precedence over the `verify_ssl` parameter.

v1.4.5 (2019-12-5)
------------------
**Changes**
 - Saving of package information can now be disabled using the `record_packages` parameter of `register_model`.

**Bugfixes**
 - Added support for uint data types to the `register_model` task.
 - Fixed an issue where long package names caused `register_model` to fail.
 - `Session` creation now works with older versions of urllib3.

v1.4.4 (2019-10-31)
-------------------
**Bugfixes**
 - Match performance definitions based on project instead of model.

v1.4.3 (2019-10-28)
-------------------
**Bugfixes**
 - Model versioning now works correctly for Python models
 - Fixed an issue where `None` values in Python caused issues with MAS models.

v1.4.2 (2019-10-23)
-------------------
**Bugfixes**
 - Fixed project properties when registering a model from ASTORE. 
 - Fixed model metadata when registering a datastep model.
 
v1.4.1 (2019-10-17)
-------------------
**Bugfixes**
 - Fixed an issue where string inputs to Python models were incorrectly handled by DS2.

v1.4 (2019-10-15)
-----------------
**Changes**
 - `PyMAS.score_code` now supports a `dest='Python'` option to retrieve the generated Python wrapper code.
 - `register_model` task includes a `python_wrapper.py` file when registering a Python model. 
 - Improved error message when user lacks required permissions to register a model. 
 
**Bugfixes**
 - Fixed an issue with CAS/EP score code that caused problems with model performance metrics.
 

v1.3 (2019-10-10)
----------------- 
**Improvements**
 - Added `update_performance` task for easily uploading performance information for a model. 
 - New (experimental) pyml2sas sub-package provides utilities for generating SAS code from Python gradient boosting models. 
 - New (experimental) methods for managing workflows added to `model_management` service.
 
**Changes**
 - `register_model` task automatically captures installed Python packages.
 - All `list_xxx` methods return all matching items unless a `limit` parameter is specified.
 - Improved API documentation.
 - Updated `full_lifecycle` example with performance monitoring.

v1.2.5 (2019-10-10)
-------------------
**Changes**
 - Registering an ASTORE model now creates an empty ASTORE file in Model Manager to be consistent with Model Studio behavior. 
 
**Bugfixes**
 - `microanalytic_score.define_steps` now works with steps having no input parameters.
 - Fixed an issue where score code generated from an ASTORE model lacked output variables.

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
