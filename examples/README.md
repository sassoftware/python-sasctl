

direct_REST_calls.py
--------------------
Level: Advanced

Demonstrates using `sasctl` to make REST calls over HTTP(S) directly to the SAS microservices.
Use if you need to customize behavior or use functionality not yet exposed through higher-level `sasctl` functions.


FleetManagement.ipynb

full_lifecycle.py

register_custom_model.py
------------------------
Level: Intermediate

Registers a model in SAS Model Manager by explicitly providing the files and model details.


Register a SAS classification model
------------------------------------
[register_sas_classification_model.py](register_sas_classification_model.py)

Level: Beginner

Registers a classification model in SAS Model Manager that was created from a SAS algorithm.

Use if you've created models using SWAT.


register_sas_dlpy_model.py
--------------------------


register_sas_regression_model.py
--------------------------------

register_scikit_classification_model.py
---------------------------------------

register_scikit_regression_model.py
-----------------------------------

Modeling in Python & SAS autoML
-------------------------------
Filename: [Viya 2020 Example.ipynb]()

Level: Intermediate

Uses the `swat` package to perform automated modeling on a dataset.  Registers the results along with a custom XGBoost model to SAS Model Manager using `sasctl`.
