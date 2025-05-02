v1.11.4 (2025-05-02)
--------------------
**Improvements**
- Improved `upload_local_model` to allow for SAS Model Manager to properly intake local ASTORE models.

v1.11.3 (2025-04-29)
--------------------
**Improvements**
- Added `upload_local_model` to `tasks.py`, which can be used to upload local directories to SAS Model Manager without any file generation.

v1.11.2 (2025-04-08)
--------------------
**Bugfixes**
- Updated `calculate_model_statistics` function in `write_json_files.py` to improve ROC tables as well as model card files.

v1.11.1 (2025-01-22)
--------------------
**Improvements**
- Improved functionality for `score_definition.py` and `score_execution.py`, allowing for more general usage of the `score_model_with_cas` task
  - Also allowed for model name to be passed into functions instead of requiring model UUID
- Pickle files now loaded with `pd.read_pickle()` instead of `pickle.load()` in score code, allowing for more flexibility between python environments

**Bugfixes**
- Updated `pzmm_generate_complete_model_card.ipynb` to have better preprocessing function

v1.11.0 (2024-10-29)
--------------------
**Changes**
- Added `score_definition.py` and `score_execution.py` to allow for score testing within SAS Model Manager
  - Included optional use of CAS Gateway for faster scoring. Only available in environments where Gateway scoring is properly set up.
- Added ability to include data pre-processing function within python score code using the `preprocess_function` argument.

**Bugfixes**
- Fixed issue where settings file was improperly imported in some score code files.

v1.10.7 (2024-10-02)
--------------------
**Changes**
 - Due to licensing restrictions, the `sasctl` package will no longer be available through Anaconda.

**Bugfixes**
 - Fixed a bug that caused an error when performing SSL verification without a CA bundle specified.

v1.10.6 (2024-08-26)
--------------------
**Improvements**
 - Refactor `tasks.py` to utilize `sasctl.pzmm` functions.
 - Add `model_info` class to better capture model information.

v1.10.5 (2024-08-01)
--------------------
**Buxfixes**
- Updated `write_json_files.py` to allow for better support for prediction models
- Fixed issues relating to model card support.

v1.10.4 (2024-07-08)
--------------------
**Improvements**
- Added example Jupyter notebook for OpenAI models.

**Buxfixes**
- Dropped support for Python 3.6 and Python 3.7, as those are no longer officially supported versions.
- Added `dmcas_misc.json` template file for model card generation.
- Updated generation of `ModelProperties.json` to allow for model card generation immediately upon upload.

v1.10.3 (2024-04-12)
--------------------
**Bugfixes**
- Updated all examples to use current versions of sasctl functions
- Fixed bug in `generate_model_card` that threw an error when trying to generate the `dmcas_misc.json` file

v1.10.2 (2024-04-10)
--------------------
**Improvements**
- Introduced `generate_model_card` into `write_json_files.py` to allow for python models to work with planned model card tab in SAS Model Manager.

**Bugfixes**
- Allow for score code to impute NaN values in tables that have been loaded into SAS Model Manager.
- Fix issue where target_value was not being properly set during score code generation
- Updated `pzmm_generate_requrirements_json.ipynb` so the requirements file is generated properly.
- Added missing statistics to `dmcas_fitstat.json` file.

v1.10.1 (2023-08-24)
--------------------
**Improvements**
- Introduced ability to specify the target index of a binary model when creating score code.
  - index can be specified in `pzmm.import_model.ImportModel.import_model()`
  - Relevant examples updated to include target_index.
  
**Bugfixes**
- Reworked `write_score_code.py` to allow for proper execution of single line scoring.
- Added template files for `assess_model_bias.py` to allow for proper execution

v1.10 (2023-08-31)
------------------
**Improvements**
 - `write_score_code.py` refactored to include ability to run batch scoring.
 - Added handling for TensorFlow Keras models.
 - Updated project creation to automatically set project properties based on contained models.
 - Included capability to assess biases of a model using CAS FairAITools using `pzmm.write_json_files.assess_model_bias()`.
 - Added custom KPI support for H2O, statsmodels, TensorFlow, and xgboost.
 - Updated examples:
   - Added example walking through the creation process of a simple TensorFlow Keras model.
   - Added example detailing the usage of `pzmm.write_json_files.assess_model_bias()` for a simple regression model
   - Updated `pzmm_custom_kpi_model_parameters` notebook to have correct parameter casing.

v1.9.4 (2023-06-15)
-------------------
**Improvements**
 - Created pytest fixture to begin running Jupyter notebooks within the GitHub automated test actions.
 - Updated examples:
   - Custom KPI and model parameters example now checks for the performance job's status.
   - Update H2O example to show model being published and scored using the "maslocal" destination.
   - Updated models to be more realistic for `pzmm_binary_classification_model_import.ipynb`.

**Bugfixes**
 - Adjust `pzmm.ScoreCode.write_score_code()` function to be compatible with future versions of pandas.
 - Reworked H2O section of `pzmm.ScoreCode.write_score_code()` to properly call H2OFrame values.
 - Fixed call to `pzmm.JSONFiles.calculate_model_statistics()` in `pzmm_binary_classification_model_import.ipynb`.

v1.9.3 (2023-06-08)
-------------------
**Improvements**
 - Refactored gitIntegration.py to `git_integration.py` and added unit tests for better test coverage.

**Bugfixes**
 - Fixed issue with ROC and Lift charts not properly being written to disk.
 - Fixed JSON conversion for Lift charts that caused TRAIN and TEST charts to be incorrect.
 - Fixed issue with H2O score code and number of curly brackets.
 - Updated score code logic for H2O to account for incompatibility with Path objects.
 - Fixed issue where inputVar.json could supply invalid values to SAS Model Manager upon model import.
 - Fixed issue with `services.model_publish.list_models`, which was using an older API format that is not valid in SAS Viya 3.5 or SAS Viya 4.

v1.9.2 (2023-05-17)
-------------------
**Improvements**
 - Add recursive folder creation and an example.
 - Add example for migrating models from SAS Viya 3.5 to SAS Viya 4.

**Bugfixes**
 - Fixed improper json encoding for `pzmm_h2o_model_import.ipynb` example.
 - Set urllib3 < 2.0.0 to allow requests to update their dependencies.
 - Set pandas >= 0.24.0 to include df.to_list alias for df.tolist.
 - Fix minor errors in h2o score code generation

v1.9.1 (2023-05-04)
-------------------
**Improvements**
 - Updated handling of H2O models in `sasctl.pzmm`.
   - Models are now saved with the appropriate `h2o` functions within the `sasctl.pzmm.PickleModel.pickle_trained_model` function.
   - Example notebooks have been updated to reflect this change.

**Bugfixes**
 - Added check for `sasctl.pzmm.JSONFiles.calculate_model_statsistics` function to replace float NaN values invalid for JSON files.
 - Fixed issue where the `sasctl.pzmm.JSONFiles.write_model_properties` function was replacing the user-defined model_function argument.
 - Added NpEncoder class to check for numpy values in JSON files. Numpy-types cannot be used in SAS Viya.

v1.9.0 (2023-04-04)
-------------------
**Improvements**
 - `sasctl.pzmm` refactored to follow PEP8 standards, include type hinting, and major expansion of code coverage.
   - `sasctl.pzmm` functions that can generate files can now run in-memory instead of writing to disk.
 - Added custom KPI handling via `pzmm.model_parameters`, allowing users to interact with the KPI table generated by model performance via API.
   - Added a method for scikit-learn models to generate hyperparameters as custom KPIs.
 - Reworked the `pzmm.write_score_code()` logic to appropriately write score code for binary classification, multi-class classification, and regression models.
 - Updated all examples based on `sasctl.pzmm` usage and model assets.
   - Examples from older versions moved to `examples/ARCHIVE/vX.X`.
 - DataStep or ASTORE models can include additional files when running `tasks.register_model()`.

**Bugfixes**
 - Fixed an issue where invalid HTTP responses could cause an error when using `Session.version_info()`.
 
v1.8.2 (2023-01-30)
-------------------
**Improvements**
 - `folders.get_folder()` can now handle folder paths and delegates (e.g. @public).

**Bugfixes**
 - Fixed an issue with `model_management.execute_model_workflow_definition()` where input values for
   workflow prompts were not correctly submitted.  Note that the `input=` parameter was renamed to
   `prompts=` to avoid conflicting with the built-in `input()`.
 - Fixed an issue with `pzmm.importModel.model_exists()` where project versions were incorrectly
   compared, resulting in improper behavior when the project version already existed.
   - Better handling for invalid project versions included.

v1.8.1 (2023-01-19)
-------------------
**Changes**
 - Adjusted workflow for code coverage reporting. Prepped to add components in next release.
 - Added `generate_requirements_json.ipynb` example.

**Bugfixes**
 - Fixed improper math.fabs use in `sasctl.pzmm.writeJSONFiles.calculateFitStat()`.
 - Fixed incorrect ast node walk for module collection in `sasctl.pzmm.writeJSONFiles.create_requirements_json()`.
 
v1.8.0 (2022-12-19)
-------------------
**Improvements**
 - Added `Session.version_info()` to check which version of Viya the session is connected to.
 - Updated the `properties=` parameter of `model_repository.create_model()` to accept a dictionary containing
   custom property names and values, and to correctly indicate their type (numeric, string, date, datetime) when
   passing the values to Viya.
 - Added `services.saslogon` for creating and removing OAuth clients.
 - Added `pzmm.JSONFiles.create_requirements_json()` to create the requirements.json file for model deployment
   to containers based on the user's model assets and Python environment.

**Changes**
 - Deprecated `core.platform_version()` in favor of `Session.version_info()`.
 - A `RuntimeError` is now raised if an obsolete service is called on a Viya 4 session (sentiment_analysis, 
   text_categorization, and text_parsing)
 - Replaced the JSON cassettes used for testing with compressed binary cassettes to save space.
 - Updated the testing framework to allow regression testing of multiple Viya versions.
 - Refactored the authentication functionality in `Session` to be more clear and less error prone.  Relevant
   functions were also made private to reduce clutter in the class's public interface.
 - Began refactor for `sasctl.pzmm` to adhere to PEP8 guidelines and have better code coverage.

**Bugfixes**
 - Fixed an issue with `register_model()` that caused invalid SAS score code to be generated when registering an
   ASTORE model in Viya 3.5.
 - Fixed a bug where calling a "get_item()" function and passing `None` would throw an error on most services instead
   of returning `None`. 
 - Fixed a bug that caused the authentication flow to be interrupted if Kerberos was missing.
 
v1.7.3 (2022-09-20)
-------------------
**Improvements**
 - Refactor astore model upload to fix 422 response from SAS Viya 4 
 - ASTORE model import now uses SAS Viya to generate ASTORE model assets
 - Expanded usage for cas_management service (credit to @SilvestriStefano)

**Bugfixes**
 - ASTORE model import no longer returns a 422 error
 - Fix improper filter usage for model_repository service
 - Fix error with loss of stream in add_model_content call for duplicate content
 - Update integration test cassettes for SAS Viya 4

v1.7.2 (2022-06-16)
-------------------
**Improvements**
 - Added a new example notebook for git integration
 - Added a model migration tool for migrating Python models from Viya 3.5 to Viya 4
 - Improved handling of CAS authentication with tokens

**Bugfixes**
 - Fixed git integration failure caused by detached head
 - Fixed minor bugs in score code generation feature
 - Fixed 500 error when importing models to Viya 4 with prewritten score code
 - Fixed incorrect handling of optional packages in pzmm

v1.7.1 (2022-04-19)
-------------------
**Bugfixes**
 - Removed linux breaking import from new git integration feature
 - Various minor bug fixes in the git integration feature

v1.7.0 (2022-04-07)
-------------------
**Improvements**
 - Added Git integration for better tracking of model history and versioning. 
 - Added MLFlow integration for simple models, allowing users to import simple MLFlow models, such as sci-kit 
   learn, to SAS Model Manager

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
