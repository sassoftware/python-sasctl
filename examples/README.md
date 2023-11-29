Examples
========
The `/examples` directory includes the examples listed below. Additionally, the `/data`
directory includes the model file outputs of the examples as well as the data sets
utilized by the examples.

Older examples are kept in the `/ARCHIVE` directory and sorted by their version of
python-sasctl.

PZMM Submodule
==============
- [Register scikit-learn binary classification models](#register-binary-classification-models)
- [Register a scikit-learn regression model](#register-a-regression-model)
- [Register a scikit-learn multiclass classification model](#register-a-multiclassification-model)
- [Register a MLFlow model](#register-a-mlflow-model)
- [Register a H2O.ai model](#register-a-h2o-model)


- [Generate a requirements.json file](#generate-a-requirements-file)
- [Create and update custom model KPIs](#create-and-update-custom-model-kpis)

---
Tasks and Services
==================
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
Register binary classification models
-------------------------------------
Filename: [pzmm_binary_classification_model_import.ipynb](pzmm_binary_classification_model_import.ipynb)

Level: Beginner

Registers a trio of classification models in SAS Model Manager that were created from a Python algorithm with [scikit-learn](https://github.com/scikit-learn/scikit-learn).


Register a regression model
---------------------------
Filename: [pzmm_regression_model_import.ipynb](pzmm_regression_model_import.ipynb)

Level: Beginner

Registers a regression model in SAS Model Manager that was created from a Python algorithm with [scikit-learn](https://github.com/scikit-learn/scikit-learn).


Register a multiclassification model
------------------------------------
Filename: [pzmm_multi_classification_model_import.ipynb](pzmm_multi_classification_model_import.ipynb)

Level: Beginner

Registers a multiclass classification model in SAS Model Manager that was created from a Python algorithm with [scikit-learn](https://github.com/scikit-learn/scikit-learn).


Register a MLFlow model
-----------------------
Filename: [pzmm_mlflow_model_import.ipynb](pzmm_mlflow_model_import.ipynb)

Level: Intermediate

Registers a classification model in SAS Model Manager that was created from a Python algorithm with [MLflow](https://github.com/mlflow/mlflow).


Register a H2O model
--------------------
Filename: [pzmm_h2o_model_import.ipynb](pzmm_h2o_model_import.ipynb)

Level: Intermediate

Registers a classification model in SAS Model Manager that was created from a Python algorithm with [H2O.ai](https://github.com/h2oai/h2o-3).


Generate a requirements file
---------------------------------
Filename: [pzmm_generate_requirements_json.ipynb](pzmm_generate_requirements_json.ipynb)

Level: Intermediate

Generates a requirements.json file which includes the minimal number of dependencies required to run a Python model


Create and update custom model KPIs
-----------------------------------
Filename: [pzmm_custom_kpis.ipynb](pzmm_custom_kpis.ipynb)

Level: Intermediate

Create and update custom model parameters and kpis on SAS Model Manager


Register a SAS classification model
------------------------------------
Filename: [register_sas_classification_model.py](register_sas_classification_model.py)

Level: Beginner

Registers a classification model in SAS Model Manager that was created from a SAS algorithm with [SWAT](https://github.com/sassoftware/python-swat).



Register a SAS regression model
-------------------------------
Filename: [register_sas_regression_model.py](register_sas_regression_model.py)

Level: Beginner

Registers a regression model in SAS Model Manager that was created from a SAS algorithm with [SWAT](https://github.com/sassoftware/python-swat).



Register a SAS deep learning model
----------------------------------
Filename: [register_sas_dlpy_model.py](register_sas_dlpy_model.py)


Level: Beginner

Creates a SAS deep learning model using [dlpy](https://github.com/sassoftware/python-dlpy) and registers the model in SAS Model Manager. (WARNING: Does not work with Python 3.10 and later)



Register a scikit-learn classification model
--------------------------------------------
Filename: [register_scikit_classification_model.py](register_scikit_classification_model.py)

Level: Beginner

Registers a classification model in SAS Model Manager that was created from a Python algorithm with [scikit-learn](https://github.com/scikit-learn/scikit-learn).



Register a scikit-learn regression model
----------------------------------------
Filename: [register_scikit_regression_model.py](register_scikit_regression_model.py)

Level: Beginner

Registers a regression model in SAS Model Manager that was created from a Python algorithm with [scikit-learn](https://github.com/scikit-learn/scikit-learn).



Full model lifecycle
--------------------
Filename: [full_lifecycle.py](full_lifecycle.py)

Level: Beginner

Demonstrates how `sasctl` can be used throughout the lifecycle of a model by:
 - training multiple Python models with [scikit-learn](https://github.com/scikit-learn/scikit-learn)
 - registering them to SAS Model Manager
 - publishing them to SAS's real-time scoring engine (MAS)
 - executing the models in real-time
 - creating a report to track model performance over time



 Register a custom model
 ------------------------
 Filename: [register_custom_model.py](register_custom_model.py)

 Level: Intermediate

 Registers a model in SAS Model Manager by explicitly providing the files and model details.



Register models with model metrics
----------------------------------
Filename: [FleetManagement.ipynb](FleetManagement.ipynb)

Level: Intermediate

Trains multiple tree-based models using [scikit-learn](https://github.com/scikit-learn/scikit-learn) and registers them in SAS Model Manager.  Also uses the `pzmm` module of `sasctl` to generate and include model fit statistics and ROC/Lift charts.



Modeling with Python & SAS AutoML
-------------------------------
Filename: [data_science_pilot.ipynb](data_science_pilot.ipynb)

Level: Intermediate

Uses the [swat](https://github.com/sassoftware/python-swat) package to perform automated modeling on a dataset.  Registers the results along with a custom XGBoost model to SAS Model Manager using `sasctl`.



Making direct REST API calls
--------------------------
Filename: [direct_REST_calls.py](direct_REST_calls.py)

Level: Advanced

Demonstrates using `sasctl` to make REST calls over HTTP(S) directly to the SAS microservices.

Use if you need to customize behavior or use functionality not yet exposed through higher-level `sasctl` functions.
