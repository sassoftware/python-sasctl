#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function
import base64
from collections import OrderedDict
import importlib
import os
import sys

try:
    import dill
    from dill import load, loads, dumps, dump
except ImportError:
    dill = None
    from pickle import load, loads, dumps, dump
import six

from .ds2 import DS2Method, DS2Thread, DS2Variable, DS2Package
from .python import ds2_variables


def build_wrapper_function(func, variables, array_input, return_msg=True):
    """Wraps a function to ensure compatibility when called by PyMAS.

    PyMAS has strict expectations regarding the format of any function called directly by PyMAS.
    Isolating the desired function inside a wrapping function provides a simple way to ensure that functions
    called by PyMAS are compliant.

    Parameters
    ----------
    func : function or str
        Function name or an instance of Function which will be wrapped
    variables : list of DS2Variable
    array_input : bool
    return_msg : bool

    Returns
    -------
    str
        the Python definition for the wrapper function.

    Notes
    -----
    The format for the `# Output: ` is very strict.  It must be exactly "# Output: <var>, <var>".  Any changes to
    spelling, capitalization, punctuation, or spacing will result in an error when the DS2 code is executed.

    """

    input_names = [v.name for v in variables if not v.out]
    output_names = [v.name for v in variables if v.out]

    args = input_names

    func = func.__name__ if callable(func) else func

    # Statement to execute the function w/ provided parameters
    if array_input:
        func_call = '{}(np.asarray({}).reshape((1,-1)))'.format(func, ','.join(args))
    else:
        func_call = '{}({})'.format(func, ','.join(args))

    # TODO: Verify that # of values returned by wrapped func matches length of output_names
    # TODO: cast all return types before returning (DS2 errors out if not exact match)

    # NOTE: 'Output:' section is required.  All return variables must be listed separated by ', '
    definition = ('def wrapper({}):'.format(', '.join(args)),
                  '    "Output: {}"'.format(', '.join(output_names + ['msg']) if return_msg
                                            else ', '.join(output_names)),
                  '    result = None',
                  '    try:',
                  '        msg = ""' if return_msg else '',
                  '        import numpy as np',
                  '        result = float({})'.format(func_call),
                  '    except Exception as e:',
                  '        msg = str(e)' if return_msg else '',
                  '        if result is None:',
                  '            result = tuple(None for i in range({}))'.format(len(output_names)),
                  '    if isinstance(result, tuple):',
                  '        return tuple(x for x in list(result) + [msg])',
                  '    else: ',
                  '        return result, msg')

    return '\n'.join(definition)


def from_inline(func, input_types=None, array_input=False, return_code=True, return_message=True):
    """Creates a PyMAS wrapper to execute the inline python function.

    Parameters
    ----------
    func : function
        A Python function object to be used
    input_types : list of type, optional
        The expected type for each input value of `func`.  Can be ommitted if `func` includes type hints.
    array_input : bool
        Whether the function inputs should be treated as an array instead of individual parameters
    return_code : bool
        Whether the DS2-generated return code should be included
    return_message : bool
        Whether the DS2-generated return message should be included

    Returns
    -------
    PyMAS
        Generated DS2 code which can be executed in a SAS scoring environment

    """

    obj = dumps(func)
    return from_pickle(obj, None, input_types, array_input, return_code, return_message)


def from_python_file(file, func_name=None, input_types=None, array_input=False, return_code=True, return_message=True):
    """ Creates a PyMAS wrapper to execute a function defined in an external .py file.

    Parameters
    ----------
    file : str
        The path to a python source code file
    func_name : str
        Name of the target function to call
    input_types : list of type, optional
        The expected type for each input value of the target function.
        Can be ommitted if target function includes type hints.
    array_input : bool
        Whether the function inputs should be treated as an array instead of individual parameters
    return_code : bool
        Whether the DS2-generated return code should be included
    return_message : bool
        Whether the DS2-generated return message should be included

    Returns
    -------
    PyMAS
        Generated DS2 code which can be executed in a SAS scoring environment

    """

    if not str(file.lower().endswith('.py')):
        raise ValueError("File {} does not have a .py extension.".format(file))

    # Extract just the filename from the path
    directory, file_name = os.path.split(file)

    # Drop the file extension to get the module name
    module_name = os.path.splitext(file_name)[0]

    # Temporarily add the module location to Python's path
    sys.path.append(directory)
    module = importlib.import_module(module_name)
    sys.path.pop()
    target_func = getattr(module, func_name)

    if not callable(target_func):
        raise RuntimeError("Could not find a valid function named {}".format(func_name))

    with open(file, 'r') as f:
        code = [line.strip('\n') for line in f.readlines()]

    return _build_pymas(target_func, None, input_types, array_input, return_code, return_message, code)


def from_pickle(file, func_name=None, input_types=None, array_input=False, return_code=True, return_message=True):
    """Create a deployable DS2 package from a Python pickle file.

    Parameters
    ----------
    file : str or bytes or file_like
        Pickled object to use.  String is assumed to be a path to a picked file, file_like is assumed to be an open
        file handle to a pickle object, and bytes is assumed to be the raw pickled bytes.
    func_name : str
        Name of the target function to call
    input_types : list of type, optional
        The expected type for each input value of the target function.
        Can be ommitted if target function includes type hints.
    array_input : bool
        Whether the function inputs should be treated as an array instead of individual parameters
    return_code : bool
        Whether the DS2-generated return code should be included
    return_message : bool
        Whether the DS2-generated return message should be included

    Returns
    -------
    PyMAS
        Generated DS2 code which can be executed in a SAS scoring environment

    """

    try:
        # In Python2 str could either be a path or the binary pickle data, so check if its a valid filepath too.
        is_file_path = isinstance(file, six.string_types) and os.path.isfile(file)
    except TypeError:
        is_file_path = False

    # Path to a pickle file
    if is_file_path:
        with open(file, 'rb') as f:
            obj = load(f)

    # The actual pickled bytes
    elif isinstance(file, bytes):
        obj = loads(file)
    else:
        obj = load(file)

    # Encode the pickled data so we can inline it in the DS2 package
    pkl = base64.b64encode(dumps(obj))

    package = 'dill' if dill else 'pickle'

    code = ('import %s, base64' % package,
            'bytes = {}'.format(pkl).replace("'", '"'),         # Replace b' with " before embedding in DS2.
            'obj = %s.loads(base64.b64decode(bytes))' % package)

    return _build_pymas(obj, func_name, input_types, array_input, return_code, return_message, code)


def _build_pymas(obj, func_name=None, input_types=None, array_input=False, return_code=True, return_message=True, code=[]):

    # If the object passed was a function, no need to search for target function
    if six.callable(obj) and (func_name is None or obj.__name__ == func_name):
        target_func = obj
    elif func_name is None:
        raise ValueError('Parameter `func_name` must be specified.')
    else:
        target_func = getattr(obj, func_name)

    if not callable(target_func):
        raise RuntimeError("Could not find a valid function named {}".format(func_name))

    # Need to create DS2Variable instances to pass to PyMAS
    if hasattr(input_types, 'columns'):
        # Assuming input is a DataFrame representing model inputs.  Use to get input variables
        vars = ds2_variables(input_types)

        # Run one observation through the model and use the result to determine output variables
        output = target_func(input_types.iloc[0, :].values.reshape((1, -1)))
        output_vars = ds2_variables(output, output_vars=True)
        vars.extend(output_vars)
    elif isinstance(input_types, type):
        params = OrderedDict([(k, input_types) for k in target_func.__code__.co_varnames])
        vars = ds2_variables(params)
    elif isinstance(input_types, dict):
        vars = ds2_variables(input_types)
    else:
        # Inspect the Python method to determine arguments
        vars = ds2_variables(target_func)

    target_func = 'obj.' + target_func.__name__

    # If all inputs should be passed as an array
    if array_input:
        first_input = vars[0]
        array_type = first_input.type or 'double'
        out_vars = [x for x in vars if x.out]
        num_inputs = len(vars) - len(out_vars)
        vars = [DS2Variable(first_input.name, array_type + '[{}]'.format(num_inputs), out=False)] + out_vars

    if not any([v for v in vars if v.out]):
        vars.append(DS2Variable(name='result', type='float', out=True))

    return PyMAS(target_func, vars, code, return_code, return_message)


class PyMAS:
    def __init__(self, target_function, variables, python_source, return_code=True, return_msg=True):
        """

        Parameters
        ----------
        target_function : function
            The Python function to be executed
        variables : list of DS2Variable
            The input/ouput variables be declared in the module
        python_source
        return_code : bool
            Whether the DS2-generated return code should be included
        return_msg : bool
            Whether the DS2-generated return message should be included

        """

        self.target = target_function

        # Any input variable that should be treated as an array
        array_input = any(v for v in variables if v.is_array)

        # Python wrapper function will serve as entrypoint from DS2
        self.wrapper = build_wrapper_function(target_function, variables,
                                              array_input, return_msg=return_msg).split('\n')

        # Lines of Python code to be embedded in DS2
        python_source = list(python_source) + list(self.wrapper)

        self.variables = variables
        self.return_code = return_code
        self.return_message = return_msg

        self.package = DS2Package()
        self.package.methods.append(DS2Method(variables, python_source))

    def score_code(self, input_table=None, output_table=None, columns=None, dest='MAS'):
        """Generate DS2 score code

        Parameters
        ----------
        input_table : str
            The name of the table to execute the function against
        output_table : str
            The name of the table where execution results will be written
        columns : list of str
            Names of the columns from `table` that will be passed to `func`
        dest : str {'MAS', 'ESP', 'CAS'}

        Returns
        -------

        """

        dest = dest.upper()

        # Check for names that could result in DS2 errors.
        DS2_KEYWORDS = ['input', 'output']
        for k in DS2_KEYWORDS:
            if input_table and k == input_table.lower():
                raise ValueError('Input table name `{}` is a reserved term.'.format(input_table))
            if output_table and k == output_table.lower():
                raise ValueError('Output table name `{}` is a reserved term.'.format(output_table))

        # Get package code
        code = (str(self.package), )

        if dest == 'ESP':
            code = ('data sasep.out;', ) + code + ('   method run();',
                                                   '      set SASEP.IN;',
                                                   '   end;',
                                                   '   method term();',
                                                   '   end;',
                                                   'enddata;')
        elif dest == 'CAS':
            if input_table is None:
                raise ValueError('An input table must be specified when executing the code in CAS.')

            output_table = output_table or input_table + '_pymas'
            thread = DS2Thread(self.variables, input_table, column_names=columns, return_message=self.return_message,
                               package=self.package)

            code += (str(thread),
                     '    data {} (overwrite=yes);'.format(output_table),
                     '    dcl thread {} t;'.format(thread.name),
                     '    method run();',
                     '       set from t;',
                     '       output {} ;'.format(output_table),
                     '    end;',
                     '  enddata;')

        return '\n'.join(code)