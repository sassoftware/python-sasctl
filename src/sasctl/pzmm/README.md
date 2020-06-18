# pickleZip-mm (pzMM) v1.0

## Overview

The goal of this package is to provide users with a straight forward method to import trained Python models into SAS Model Manager and SAS Open Model Manager. In order to complete this task, the package completes the following:

* Writes `.JSON` files needed for SAS Open Model Manager to read in the model information, which includes the following files:
  * fileMetadata.json specifies the file roles for the names of the input and output variables files, the Python score code file, and the Python pickle file
  * ModelProperties.json is used to set the model properties that are read by SAS Open Model Manager during the import process
  * inputVar.json and outputVar.json are used to set the input and output variables to be used by OMM
  * dmcas_fitstat.json is an optional file that provides the fit statistics that are associated with the imported model, which are either user-generated or data-generated
  * dmcas_lift.json and dmcas_roc.json are optional files that provide the lift and ROC plots that are associated with the imported model, which are data-generated
* Writes the *score.py file that is used for model scoring
* Serializes a trained model in to a binary pickle file
* Archives all relevant model files in to a ZIP file and imports the model using REST API calls

## Prerequisites

Use of this package requires the following:

* Python version 3+
* SAS Viya 3.5+ environment and user credentials
* External Python libraries:
  * scipy v1.4.0
  * scikit v0.22.1
  * pandas v0.25.3
  * requests v2.23.0

## Installation

In order to install this package, run the following command:

```bash
git clone <SSH or HTTPS>
```

Also, when in the parent directory `~/picklezip-mm/`, run the following command:

```bash
pip install .
```

In order to upgrade the package after a git pull request, simply use the above command again, which uninstalls the old version and installs the new version in its place.

To completely uninstall the package, run the following command:

```bash
pip uninstall picklezip-mm
```

## Getting Started

The easiest way to get started with the picklezip-mm package is to follow the ImportPythonModel.ipynb notebook in the [examples](/examples) directory. Example data sets and an importable model ZIP file is included in the [samples](/samples/Python_Models/DTree_sklearn_PyPickleModel) directory.


## License

This project is licensed under the [Apache 2.0 License](/LICENSE).


