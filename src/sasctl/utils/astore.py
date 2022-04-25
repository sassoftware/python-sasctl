#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import shutil
import tempfile
import uuid
import zipfile


try:
    import swat
except ImportError:
    swat = None


def create_package(table, input=None):
    """Create an importable model package from a CAS table.

    Parameters
    ----------
    table : swat.CASTable
        The CAS table containing an ASTORE or score code.
    input : DataFrame, type, list of type, or dict of str: type, optional
        The expected type for each input value of the target function.
        Can be omitted if target function includes type hints.  If a DataFrame
        is provided, the columns will be inspected to determine type information.
        If a single type is provided, all columns will be assumed to be that type,
        otherwise a list of column types or a dictionary of column_name: type
        may be provided.

    Returns
    -------
    BytesIO
        A byte stream representing a ZIP archive which can be imported.

    See Also
    --------
    :meth:`model_repository.import_model_from_zip <.ModelRepository.import_model_from_zip>`

    """
    if swat is None:
        raise RuntimeError("The 'swat' package is required to work with SAS models.")

    if not isinstance(table, swat.CASTable):
        raise ValueError(
            "Parameter 'table' should be an instance of '%r' but "
            "received '%r'." % (swat.CASTable, table)
        )

    if "DataStepSrc" in table.columns:
        # Input only passed to datastep
        return create_package_from_datastep(table, input=input)
    return create_package_from_astore(table)


def create_package_from_datastep(table, input=None):
    """Create an importable model package from a score code table.

    Parameters
    ----------
    table : swat.CASTable
        The CAS table containing the score code.
    input : DataFrame, type, list of type, or dict of str: type, optional
        The expected type for each input value of the target function.
        Can be omitted if target function includes type hints.  If a DataFrame
        is provided, the columns will be inspected to determine type information.
        If a single type is provided, all columns will be assumed to be that type,
        otherwise a list of column types or a dictionary of column_name: type
        may be provided.

    Returns
    -------
    BytesIO
        A byte stream representing a ZIP archive which can be imported.

    See Also
    --------
    :meth:`model_repository.import_model_from_zip <.ModelRepository.import_model_from_zip>`

    """
    dscode = table.to_frame().loc[0, "DataStepSrc"]

    # Extract inputs if provided
    input_vars = []
    # Workaround because sasdataframe does not like to be check if exist
    if str(input) != "None":
        from .pymas.python import ds2_variables

        variables = None
        if hasattr(input, "columns"):
            # Assuming input is a DataFrame representing model inputs.  Use to
            # get input variables
            variables = ds2_variables(input)
        elif isinstance(input, dict):
            variables = ds2_variables(input)
        if variables:
            input_vars = [v.as_model_metadata() for v in variables if not v.out]

    # Find outputs from ds code
    output_vars = []
    for sasline in dscode.split("\n"):
        if sasline.strip().startswith("label"):
            output_var = {}
            for tmp in sasline.split("="):
                if "label" in tmp:
                    ovarname = tmp.split("label")[1].strip()
                    output_var.update({"name": ovarname})
                    # Determine type of variable is decimal or string
                    if "length " + ovarname in dscode:
                        sastype = (
                            dscode.split("length " + ovarname)[1].split(";")[0].strip()
                        )
                        if "$" in sastype:
                            output_var.update({"type": "string"})
                            output_var.update({"length": sastype.split("$")[1]})
                        else:
                            output_var.update({"type": "decimal"})
                            output_var.update({"length": sastype})
                    else:
                        # If no length for variable, default is decimal, 8
                        output_var.update({"type": "decimal"})
                        output_var.update({"length": 8})
                else:
                    output_var.update(
                        {"description": tmp.split(";")[0].strip().strip("'")}
                    )
            output_vars.append(output_var)

    file_metadata = [{"role": "score", "name": "dmcas_scorecode.sas"}]

    zip_file = _build_zip_from_files(
        {
            "fileMetadata.json": file_metadata,
            "dmcas_scorecode.sas": dscode,
            "ModelProperties.json": {"scoreCodeType": "dataStep"},
            "outputVar.json": output_vars,
            "inputVar.json": input_vars,
        }
    )

    return zip_file


def create_package_from_astore(table):
    """Create an importable model package from an ASTORE.

    Parameters
    ----------
    table : swat.CASTable
        The CAS table containing the ASTORE.

    Returns
    -------
    BytesIO
        A byte stream representing a ZIP archive which can be imported.

    See Also
    --------
    :meth:`model_repository.import_model_from_zip <.ModelRepository.import_model_from_zip>`

    """
    files = create_files_from_astore(table)

    return _build_zip_from_files(files)


def create_files_from_astore(table):
    """Generate files for importing a model from an ASTORE.

    Parameters
    ----------
    table : swat.CASTable
        The CAS table containing the ASTORE.

    Returns
    -------
    dict
        Dictionary of filename: content pairs.

    """
    if swat is None:
        raise RuntimeError(
            "The 'swat' package is required to work with " "ASTORE models."
        )

    if not isinstance(table, swat.CASTable):
        raise ValueError(
            "Parameter 'table' should be an instance of '%r' but "
            "received '%r'." % (swat.CASTable, table)
        )

    sess = table.session.get_connection()
    sess.loadactionset("astore")

    result = sess.astore.describe(rstore=table, epcode=True)

    # Model Manager expects a 0-byte ASTORE file.  Will retrieve actual ASTORE
    # from CAS during model publish.
    astore = bytes()

    # Raise error if describe action fails
    if result.status_code != 0:
        raise RuntimeError(result)

    astore_key = result.Key.Key[0].strip()

    # Remove "Keep" sas code from CAS/EP code so full table plus output are
    # returned. This is so the MM performance charts and test work.
    keepstart = result.epcode.find("Keep")
    keepend = result.epcode.find(";", keepstart)
    ep_ds2 = result.epcode[0:keepstart] + result.epcode[keepend + 1 :]

    package_ds2 = _generate_package_code(result)
    model_properties = _get_model_properties(result)
    input_vars = [
        get_variable_properties(var) for var in result.InputVariables.itertuples()
    ]
    input_vars = [v for v in input_vars if v.get("role", "").upper() == "INPUT"]
    output_vars = [
        get_variable_properties(var) for var in result.OutputVariables.itertuples()
    ]
    astore_filename = "_" + uuid.uuid4().hex[:25].upper()

    # Copy the ASTORE table to the ModelStore.
    # Raise an error if the action fails
    with swat.options(exception_on_severity=2):
        table.save(name=astore_filename, caslib="ModelStore", replace=True)

    file_metadata = [
        {"role": "analyticStore", "name": ""},
        {"role": "score", "name": "dmcas_epscorecode.sas"},
    ]

    astore_metadata = [
        {
            "name": astore_filename,
            "caslib": "ModelStore",
            "uri": "/dataTables/dataSources/cas~fs~cas-shared-default~fs~ModelStore/tables/{}".format(
                astore_filename
            ),
            "key": astore_key,
        }
    ]

    return {
        "dmcas_packagescorecode.sas": "\n".join(package_ds2),
        "dmcas_epscorecode.sas": ep_ds2,
        astore_filename: astore,
        "ModelProperties.json": model_properties,
        "fileMetadata.json": file_metadata,
        "AstoreMetadata.json": astore_metadata,
        "inputVar.json": input_vars,
        "outputVar.json": output_vars,
    }


def _build_zip_from_files(files):
    """Create a ZIP file containing the provided files.

    Parameters
    ----------
    files : dict
        Dictionary of filename: content to be added to the .zip file.

    Returns
    -------
    BytesIO
        Byte stream representation of the .zip file.

    """
    try:
        # Create a temp folder
        folder = tempfile.mkdtemp()

        for k, v in files.items():
            filename = os.path.join(folder, k)

            # Write JSON file
            if os.path.splitext(k)[-1].lower() == ".json":
                with open(filename, "w") as f:
                    json.dump(v, f, indent=1)
            else:
                mode = "wb" if isinstance(v, bytes) else "w"

                with open(filename, mode) as f:
                    f.write(v)

        files = os.listdir(folder)

        with zipfile.ZipFile(os.path.join(folder, "model.zip"), "w") as z:
            for file in files:
                z.write(os.path.join(folder, file), file)

        # Need to return the ZIP file data but also need to ensure the
        # directory is cleaned up.
        # Read the bytes from disk and return an in memory "file".
        with open(os.path.join(folder, "model.zip"), "rb") as z:
            return io.BytesIO(z.read())
    finally:
        shutil.rmtree(folder)


def get_variable_properties(var):
    type_mapping = {"interval": "", "num": "decimal", "character": "string"}

    meta = {"name": var.Name.strip(), "length": int(var.Length)}

    # Input variable table has Type & RawType columns, but RawType aligns with Type column from Output variable table.
    if hasattr(var, "RawType"):
        meta["type"] = type_mapping[var.RawType.strip().lower()]
    else:
        meta["type"] = type_mapping[var.Type.strip().lower()]

    if hasattr(var, "Role"):
        meta["role"] = var.Role.strip().upper()

    return meta


def _get_model_properties(result):
    properties = {
        "custom properties": [],
        "externalUrl": "",
        "trainTable": "",
        "trainCodeType": "",
        "description": "",
        "tool": "SAS Visual Data Mining and Machine Learning",
        "toolVersion": "",
        "targetVariable": "",
        "scoreCodeType": "ds2MultiType",
        "externalModelId": "",
        "function": "",
        "eventProbVar": "",
        "modeler": "",
        "name": "",
        "targetEvent": "",
        "targetLevel": "",
        "algorithm": "",
    }

    algorithm = result.Description[result.Description.Attribute == "Analytic Engine"]
    if algorithm.size > 0:
        algorithm = str(algorithm.Value.iloc[0]).lower()
    else:
        algorithm = None

    def is_classification(r):
        """Determine if the ASTORE model describes a classification model."""
        return classification_target(r) is not None

    def classification_target(r):
        """Get the name of the classification target variable."""
        target = r.OutputVariables.Name[r.OutputVariables.Name.str.startswith("I_")]
        if target.shape[0] > 0:
            return target.iloc[0].replace("I_", "", 1)
        return None

    def regression_target(r):
        """Get the name of the regression target variable."""
        target = r.OutputVariables.Name.str.startswith("P_")
        target = r.OutputVariables.Name[target].iloc[0]
        return target.replace("P_", "", 1)

    if algorithm == "glm":
        properties["algorithm"] = "Linear regression"
        properties["tool"] = "SAS Visual Analytics"
        properties["function"] = "prediction"
        properties["targetVariable"] = regression_target(result)

    elif algorithm == "logistic":
        properties["algorithm"] = "Logistic regression"
        properties["tool"] = "SAS Visual Analytics"
        properties["function"] = "classification"
        properties["targetVariable"] = classification_target(result)

    elif algorithm == "forest":
        properties["algorithm"] = "Random forest"

        if is_classification(result):
            properties["function"] = "classification"
            properties["targetVariable"] = classification_target(result)
        else:
            properties["function"] = "prediction"
            properties["targetVariable"] = regression_target(result)

    elif algorithm == "gradboost":
        properties["algorithm"] = "Gradient boosting"

        if is_classification(result):
            properties["function"] = "classification"
            properties["targetVariable"] = classification_target(result)

            if result.OutputVariables.Name.str.startswith("P_").sum() == 2:
                properties["targetLevel"] = "binary"
        else:
            properties["function"] = "prediction"
            properties["targetVariable"] = regression_target(result)

    elif algorithm == "svmachine":
        properties["algorithm"] = "Support vector machine"

        if is_classification(result):
            properties["function"] = "classification"
            properties["targetVariable"] = classification_target(result)
            properties["targetLevel"] = "binary"
        else:
            properties["function"] = "prediction"
            properties["targetVariable"] = regression_target(result)

    elif algorithm == "bnet":
        properties["algorithm"] = "Bayesian network"
        properties["function"] = "classification"
        properties["targetVariable"] = classification_target(result)

        if result.OutputVariables.Name.str.startswith("P_").sum() == 2:
            properties["targetLevel"] = "binary"

    else:
        properties["tool"] = ""

    return properties


def _generate_package_code(result):
    """Generates package-style DS2 code from EP-style DS2 code."""

    id_ = "_" + uuid.uuid4().hex  # Random ID for package
    key = result.Key.Key[0]

    header = (
        "package ds2score / overwrite=yes;",
        "    dcl package score {}();".format(id_),
    )

    dcl_lines = []
    for line in result.epcode.split("\n"):
        # Ignore the package declaration since it will be redefined
        if line.strip().startswith("dcl ") and not line.strip().startswith(
            "dcl package "
        ):
            dcl_lines.append(line)

    init_method = (
        "    varlist allvars [_all_];",
        " ",
        "    method init();",
        "       {}.setvars(allvars);".format(id_),
        "       {}.setkey(n'{}');".format(id_, key),
        "    end;",
    )

    def extract_type(var, out=False):
        # Find the matching variable declarations and extract the type
        var = str(var).strip()
        x = [x for x in dcl_lines if ' "{}"'.format(var) in x][0]
        x = x.replace("dcl ", "").strip().split(" ")[0]

        # Remove the length component from output variables to prevent
        # compilation warning which prevents publishing to MAS
        if out and "(" in x:
            x = x[: x.find("(")]
        return x

    variables = []
    # Despite being call "InputVariables" at least some ASTORE models
    # include the target variable in the list
    for _, row in result.InputVariables.iterrows():
        if "Role" in row and row["Role"].lower() != "target":
            name = row["Name"]
            variables.append('       %s "%s"' % (extract_type(name), name))
    variables += [
        '       IN_OUT {} "{}"'.format(extract_type(var, out=True), var)
        for var in result.OutputVariables.Name
    ]

    score_method = ("    method score(", ",\n".join(variables), "   );")
    score_method += tuple(
        '       this."{var}" = "{var}";'.format(var=v)
        for v in result.InputVariables.Name
    )
    score_method += (" ", "       {}.scorerecord();".format(id_), " ")
    score_method += tuple(
        '       "{var}" = this."{var}";'.format(var=v)
        for v in result.OutputVariables.Name
    )

    footer = ("    end;", "endpackage;")

    return header + tuple(dcl_lines) + init_method + score_method + footer
