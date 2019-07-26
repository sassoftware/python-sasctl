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


def create_package_from_astore(table):
    if swat is None:
        raise RuntimeError("The 'swat' package is required to work with ASTORE models.")

    assert isinstance(table, swat.CASTable)

    sess = table.session.get_connection()
    sess.loadactionset('astore')

    result = sess.astore.describe(rstore=table, epcode=True)
    astore = sess.astore.download(rstore=table).blob
    astore = bytes(astore)      # Convert from SWAT blob type

    # Raise error if describe action fails
    if result.status_code != 0:
        raise RuntimeError(result)

    astore_key = result.Key.Key[0].strip()
    ds2 = _generate_package_code(result)
    model_properties = _get_model_properties(result)
    input_vars = [get_variable_properties(var) for var in result.InputVariables.itertuples()]
    output_vars = [get_variable_properties(var) for var in result.OutputVariables.itertuples()]
    astore_filename = '_' + uuid.uuid4().hex[:25].upper()

    # Copy the ASTORE table to the ModelStore.
    # Raise an error if the action fails
    with swat.options(exception_on_severity=2):
        table.save(name=astore_filename, caslib='ModelStore', replace=True)

    file_metadata = [{'role': 'analyticStore', 'name': ''},
                     {'role': 'score', 'name': 'dmcas_packagescorecode.sas'}]

    astore_metadata = [{'name': astore_filename,
                        'caslib': 'ModelStore',
                        'uri': '/dataTables/dataSources/cas~fs~cas-shared-default~fs~ModelStore/tables/{}'.format(astore_filename),
                        'key': astore_key}]

    try:
        # Create a temp folder
        folder = tempfile.mkdtemp()

        # Closure for easily adding JSON files
        def json_file(data, filename):
            filename = os.path.join(folder, filename)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=1)

        filename = os.path.join(folder, 'dmcas_packagescorecode.sas')
        with open(filename, 'w') as f:
            f.write('\n'.join(ds2))

        filename = os.path.join(folder, astore_filename)
        with open(filename, 'wb') as f:
            f.write(astore)

        json_file(model_properties, 'ModelProperties.json')
        json_file(file_metadata, 'fileMetadata.json')
        json_file(astore_metadata, 'AstoreMetadata.json')
        json_file(input_vars, 'inputVar.json')
        json_file(output_vars, 'outputVar.json')

        files = os.listdir(folder)

        with zipfile.ZipFile(os.path.join(folder, 'model.zip'), 'w') as z:
            for file in files:
                z.write(os.path.join(folder, file), file)

        # Need to return the ZIP file data but also need to ensure the directory is cleaned up.
        # Read the bytes from disk and return an in memory "file".
        with open(os.path.join(folder, 'model.zip'), 'rb') as z:
            return io.BytesIO(z.read())

    finally:
        shutil.rmtree(folder)


def get_variable_properties(var):
    type_mapping = {
        'interval': '',
        'num': 'decimal',
        'character': 'string'
    }

    meta = {'name': var.Name.strip(), 'length': int(var.Length)}

    # Input variable table has Type & RawType columns, but RawType aligns with Type column from Output variable table.
    if hasattr(var, 'RawType'):
        meta['type'] = type_mapping[var.RawType.strip().lower()]
    else:
        meta['type'] = type_mapping[var.Type.strip().lower()]

    if hasattr(var, 'Role'):
        meta['role'] = var.Role.strip().upper()

    return meta


def _get_model_properties(result):
    return {
        "custom properties": [],
        "externalUrl": "",
        "trainTable": "",
        "trainCodeType": "",
        "description": "",
        "tool": "",
        "toolVersion": "",
        "targetVariable": "",
        "scoreCodeType": "ds2Package",
        "externalModelId": "",
        "function": "",
        "eventProbVar": "",
        "modeler": "",
        "name": "CustomerLifetimeValueScore",
        "targetEvent": "",
        "targetLevel": "",
        "algorithm": ""
    }


def _generate_package_code(result):
    """Generates package-style DS2 code from EP-style DS2 code."""

    id = '_' + uuid.uuid4().hex  # Random ID for package
    key = result.Key.Key[0]

    header = ('package ds2score / overwrite=yes;',
              '    dcl package score {}();'.format(id))

    dcl_lines = []
    for line in result.epcode.split('\n'):
        # Ignore the package declaration since it will be redefined
        if line.strip().startswith('dcl ') and not line.strip().startswith('dcl package '):
            dcl_lines.append(line)

    init_method = ('    varlist allvars [_all_];',
                   ' ',
                   '    method init();',
                   "       {}.setvars(allvars);".format(id),
                   "       {}.setkey(n'{}');".format(id, key),
                   '    end;')

    def extract_type(var, out=False):
        # Find the matching variable declarations and extract the type
        var = str(var).strip()
        l = [l for l in dcl_lines if ' "{}"'.format(var) in l][0]
        l = l.replace('dcl ', '').strip().split(' ')[0]

        # Remove the length component from output variables
        # Otherwise, compilation warning is raised which prevents publishing to MAS
        if out and '(' in l:
            l = l[:l.find('(')]
        return l


    variables = ['       {} "{}"'.format(extract_type(var), var) for var in result.InputVariables.Name]
    variables += ['       IN_OUT {} "{}"'.format(extract_type(var, out=True), var) for var in result.OutputVariables.Name]

    score_method = ('    method score(',
                    ',\n'.join(variables),
                    '   );')
    score_method += tuple('       this."{}" = "{}";'.format(var, var) for var in result.InputVariables.Name)
    score_method += (' ',
                     '       {}.scorerecord();'.format(id),
                     ' ')
    score_method += tuple('       "{}" = this."{}";'.format(var, var) for var in result.InputVariables.Name)

    footer = ('    end;',
              'endpackage;')

    return header + tuple(dcl_lines) + init_method + score_method + footer

