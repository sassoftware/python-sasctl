
Unreleased
----------
**Improvements**
 - public_model task also defines methods mapped to MAS module steps when publishing to MAS.
 - SSL verification can be disable with `SSLREQCERT` environment variable.
 - Added `execute_performance_task`

**Changes**
 - Updated method signature for `create_performance_definition` in Model Manager.

**Bugfixes**
 - register_model task no longer adds `rc` and `msg` variables from MAS to the project variables.

v0.9.6 (2019-07-15)
-------------------
Initial public release.