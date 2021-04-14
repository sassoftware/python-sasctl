
Examples
========

- [Register a SAS classification model](#register-a-sas-classification-model)
- [Register a SAS regression model](#register-a-sas-regression-model)
- [Register a SAS deep learning model](#register-a-sas-deep-learning-model)

- [Register a scikit-learn classification model](#register-a-scikit-learn-classification-model)
- [Register a scikit-learn regression model](#register-a-scikit-learn-regression-model)

- [Full model lifecycle](#full-model-lifecycle)
- [Register a custom model](#register-a-custom-model)
- [Register models with model metrics](#register-models-with-model-metrics)
- [Modeling with Python & SAS AutoML](#modeling-with-python--sas-automl)
- [Making direct REST API calls](#making-direct-rest-api-calls)

---


Register a SAS classification model
------------------------------------
[register_sas_classification_model.py](register_sas_classification_model.py)

Level: Beginner

Registers a classification model in SAS Model Manager that was created from a SAS algorithm with [SWAT](https://github.com/sassoftware/python-swat).



Register a SAS regression model
-------------------------------
[register_sas_regression_model.py](register_sas_regression_model.py)

Level: Beginner

Registers a regression model in SAS Model Manager that was created from a SAS algorithm with [SWAT](https://github.com/sassoftware/python-swat).



Register a SAS deep learning model
----------------------------------
[register_sas_dlpy_model.py](register_sas_dlpy_model.py)

Level: Beginner

Creates a SAS deep learning model using [dlpy](https://github.com/sassoftware/python-dlpy) and registers the model in SAS Model Manager.



Register a scikit-learn classification model
--------------------------------------------
[register_scikit_classification_model.py](register_scikit_classification_model.py)

Level: Beginner

Registers a classification model in SAS Model Manager that was created from a Python algorithm with [scikit-learn](https://github.com/scikit-learn/scikit-learn).



Register a scikit-learn regression model
----------------------------------------
[register_scikit_regression_model.py](register_scikit_regression_model.py)

Level: Beginner

Registers a regression model in SAS Model Manager that was created from a Python algorithm with [scikit-learn](https://github.com/scikit-learn/scikit-learn).



Full model lifecycle
--------------------
[full_lifecycle.py](full_lifecycle.py)

Level: Beginner

Demonstrates how `sasctl` can be used throughout the lifecycle of a model by:
 - training multiple Python models with [scikit-learn](https://github.com/scikit-learn/scikit-learn)
 - registering them to SAS Model Manager
 - publishing them to SAS's real-time scoring engine (MAS)
 - executing the models in real-time
 - creating a report to track model performance over time



 Register a custom model
 ------------------------
 [register_custom_model.py](register_custom_model.py)

 Level: Intermediate

 Registers a model in SAS Model Manager by explicitly providing the files and model details.



Register models with model metrics
----------------------------------
[FleetManagement.ipynb](FleetManagement.ipynb)

Level: Intermediate

Trains multiple tree-based models using [scikit-learn](https://github.com/scikit-learn/scikit-learn) and registers them in SAS Model Manager.  Also uses the `pzmm` module of `sasctl` to generate and include model fit statistics and ROC/Lift charts.



Modeling with Python & SAS AutoML
-------------------------------
Filename: [Viya 2020 Example.ipynb]()

Level: Intermediate

Uses the [swat](https://github.com/sassoftware/python-swat) package to perform automated modeling on a dataset.  Registers the results along with a custom XGBoost model to SAS Model Manager using `sasctl`.



Making direct REST API calls
--------------------------
[direct_REST_calls.py](direct_REST_calls.py)

Level: Advanced

Demonstrates using `sasctl` to make REST calls over HTTP(S) directly to the SAS microservices.

Use if you need to customize behavior or use functionality not yet exposed through higher-level `sasctl` functions.
