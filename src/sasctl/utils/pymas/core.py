#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contains utilities for wrapping Python models in DS2 for publishing."""

from __future__ import print_function
import base64
import gzip
import importlib
import pickle  # skipcq BAN-B301
import os
import re
import sys
from collections import OrderedDict

from .ds2 import DS2Thread, DS2Variable, DS2PyMASPackage
from .python import ds2_variables
from ..decorators import versionadded, versionchanged
from ..misc import random_string


@versionchanged(reason='Added `name` parameter.', version='1.5')
def build_wrapper_function(
    func, variables, array_input, name='wrapper', setup=None, return_msg=None
):
    """Wraps a function to ensure compatibility when called by PyMAS.

    PyMAS has strict expectations regarding the format of any function called
    directly by PyMAS.  Isolating the desired function inside a wrapping
    function provides a simple way to ensure that functions called by PyMAS
    are compliant.

    Parameters
    ----------
    func : function or str
        Function name or an instance of Function which will be wrapped
    variables : list of DS2Variable
    array_input : bool
        Whether `variables` should be combined into a single array before
        passing to `func`
    name : str, optional
        Name of the generated wrapper function.  Defaults to 'wrapper'.
    setup : iterable
        Python source code lines to be executed during package setup
    return_msg : bool, optional
        Deprecated.
        Whether error messages will be captured and returned as an additional
        output variable.  Defaults to True.

    Returns
    -------
    str
        the Python definition for the wrapper function.

    Notes
    -----
    The format for the `# Output: ` is very strict.  It must be exactly
    "# Output: <var>, <var>".  Any changes to spelling, capitalization,
    punctuation, or spacing will result in an error when the DS2 code is
    executed.

    An additional return variable named 'msg' is always added.  Since the expected use of this function
    is to generate Python methods that will be called from DS2, this variable will contain a stack trace in
    the event an error occurs, allowing it to be logged/exposed by DS2.

    """
    if return_msg is not None:
        raise DeprecationWarning(
            "The 'return_msg' parameter is ignored and "
            "will be removed completely in a future "
            "version"
        )

    # Always return an error message out of any Python method call.
    return_msg = True

    # need to know func & input/output vars for each func
    input_names = [v.name for v in variables if not v.out]
    output_names = [v.name for v in variables if v.out]
    args = input_names
    func = func.__name__ if callable(func) else func

    # HELPER: SAS to python char issue where SAS char have spaces and python string does not.
    # NOTE: we assume SAS char always need white space to be trimmed.  This seems to match python model built so far
    # Note: Using this string fix to help with None or blank input situation for numerics
    string_input = ('',)
    for v in variables:
        if not v.out:
            if v.type == 'char':
                string_input += ("        if {0}: {0} = {0}.strip()".format(v.name),)
            else:
                string_input += ("        if {0} is None: {0} = np.nan".format(v.name),)

    # Statement to execute the function w/ provided parameters
    if array_input:
        middle = string_input + (
            '        input_array = np.array([{}]).reshape((1, -1))'.format(
                ', '.join(args)
            ),
            '        columns = [{}]'.format(', '.join('"{0}"'.format(w) for w in args)),
            '        input_df = pd.DataFrame(data=input_array, columns=columns)',
            '        result = {}(input_df)'.format(func),
        )
    else:
        func_call = '{}({})'.format(func, ','.join(args))
        middle = ('        result = {}'.format(func_call),)

    if setup:
        header = (
            ('try:',)
            + tuple('    ' + line for line in setup)
            + (
                '    _compile_error = None',
                'except Exception as e:',
                '    _compile_error = e',
                '',
            )
        )
    else:
        header = ('',)

    # NOTE: 'Output:' section is required.  All return variables must be listed
    # separated by ', '
    definition = (
        header
        + (
            'def {name}({args}):'.format(name=name, args=', '.join(args)),
            '    "Output: {}"'.format(
                ', '.join(output_names + ['msg'])
                if return_msg
                else ', '.join(output_names)
            ),
            '    result = None',
            '    msg = None' if return_msg else '',
            '    try:',
            '        global _compile_error',
            '        if _compile_error is not None:',
            '            raise _compile_error',
            '        import numpy as np',
            '        import pandas as pd',
        )
        + middle
        + (
            '        result = tuple(result.ravel()) if hasattr(result, '
            '"ravel") else tuple(result)',
            '        if len(result) == 0:',
            '            result = tuple(None for i in range(%s))' % len(output_names),
            '        elif "numpy" in str(type(result[0])):',
            '            result = tuple(np.asscalar(i) for i in result)',
            '    except Exception as e:',
            '        from traceback import format_exc',
            '        msg = str(e) + format_exc()' if return_msg else '',
            '        if result is None:',
            '            result = tuple(None for i in range(%s))' % len(output_names),
            '    return result + (msg, )',
        )
    )

    return '\n'.join(definition)


@versionadded(version='1.5')
def wrap_predict_method(func, variables, **kwargs):
    """Create a PyMAS wrapper designed for Scikit's `.predict` methods.

    Parameters
    ----------
    func : function or str
        Function name or an instance of Function which will be wrapped.  Assumed
        to behave as `.predict()` methods.
    variables : list of DS2Variable
        Input and output variables for the function
    kwargs : any
        Will be passed to `build_wrapper_function`.

    Returns
    -------
    str
        the Python definition for the wrapper function.

    See Also
    --------
    :meth:`build_wrapper_function`

    """
    kwargs.setdefault('array_input', True)
    kwargs.setdefault('name', 'predict')

    return build_wrapper_function(func, variables, **kwargs)


@versionadded(version='1.5')
def wrap_predict_proba_method(func, variables, **kwargs):
    """Create a PyMAS wrapper designed for Scikit's `.predict_proba` methods.

    Parameters
    ----------
    func : function or str
        Function name or an instance of Function which will be wrapped.  Assumed
        to behave as `.predict_proba()` methods.
    variables : list of DS2Variable
        Input and output variables for the function
    kwargs : any
        Will be passed to `build_wrapper_function`.

    Returns
    -------
    str
        the Python definition for the wrapper function.

    See Also
    --------
    :meth:`build_wrapper_function`

    """
    kwargs.setdefault('array_input', True)
    kwargs.setdefault('name', 'predict_proba')

    wrapper = build_wrapper_function(func, variables, **kwargs)

    # Expecting output to be ndarray of probabilities instead of single value
    old_code = r'if result.size == 1:\s*result = np.asscalar\(result\)'
    new_code = 'assert result.shape[0] == 1\n'
    new_code += '        result = tuple(result[0].tolist())'

    return re.sub(old_code, new_code, wrapper)


@versionadded(version='1.6')
def wrap_classification_score_method(func, variables, class_names, event_level=None, **kwargs):
    """

    Parameters
    ----------
    func
    variables : list of DS2Variable
    class_names : list of string
    event_level : int
    kwargs : any

    Returns
    -------
    str

    """
    kwargs.setdefault('array_input', True)
    kwargs.setdefault('name', 'score')

    # Make sure the extra output variables expected by SAS are present
    if not any(v.name == 'EM_CLASSIFICATION' for v in variables):
        variables.extend([
            DS2Variable('I_Target', 'str', True),
            DS2Variable('EM_CLASSIFICATION', 'str', True),
            DS2Variable('EM_PROBABILITY', 'double', True),
            DS2Variable('EM_EVENTPROBABILITY', 'double', True),
        ])

    wrapper_code = build_wrapper_function(func, variables, **kwargs)

    # Need to find position of indent before 'except' block.
    match = re.search(r'\s*except Exception', wrapper_code)

    if event_level is None:
        event_level = len(class_names) - 1

    new_code = """
        event_index = {event}
        class_names = {classes}
        class_index = result.index(max(result))

        I_Target = class_names[class_index]
        EM_CLASSIFICATION = I_Target
        EM_PROBABILITY = result[class_index]
        EM_EVENTPROBABILITY = result[event_index]

        result += (I_Target, EM_CLASSIFICATION, EM_EVENTPROBABILITY, EM_PROBABILITY)
    """.format(event=event_level,
               classes=str(class_names).replace("'", '"'))

    wrapper_code = wrapper_code[:match.start()] + new_code + wrapper_code[match.start():]
    return wrapper_code


@versionadded(version='1.6')
def wrap_regression_score_method(func, variables, **kwargs):

    # EM_PREDICTION
    # P_MORTDUE (P_TargetName)

    kwargs.setdefault('array_input', True)
    kwargs.setdefault('name', 'score')

    # Make sure the extra output variables expected by SAS are present
    if not any(v.name == 'EM_PREDICTION' for v in variables):
        # There should be exactly 1 output variable defined at this point. (multiple regression current not handled)
        output_var = [v for v in variables if v.out]

        if len(output_var) != 1:
            raise ValueError('Unable to find an output variable.')
        output_var = output_var[0]
        variables.remove(output_var)

        variables.extend([
            DS2Variable('P_%s' % output_var.name, 'double', True),
            DS2Variable('EM_PREDICTION', 'double', True)
        ])

    wrapper_code = build_wrapper_function(func, variables, **kwargs)

    # Need to find position of indent before 'except' block.
    match = re.search(r'\s*except Exception', wrapper_code)

    # Have 2 output variables to provide (same) output for, but model only outputs 1 value
    # Since output will have been converted to tuple, can simply replicate values.
    new_code = """
        result *= 2
    """

    wrapper_code = wrapper_code[:match.start()] + new_code + wrapper_code[match.start():]
    return wrapper_code


def from_inline(
    func, input_types=None, array_input=False
):
    """Creates a PyMAS wrapper to execute the inline python function.

    Parameters
    ----------
    func : function
        A Python function object to be used
    input_types : list of type, optional
        The expected type for each input value of `func`.  Can be ommitted if
        `func` includes type hints.
    array_input : bool
        Whether the function inputs should be treated as an array instead of
        individual parameters

    Returns
    -------
    PyMAS
        Generated DS2 code which can be executed in a SAS scoring environment

    """

    obj = pickle.dumps(func)
    return from_pickle(obj, None, input_types, array_input)


def from_python_file(
    file,
    func_name=None,
    input_types=None,
    array_input=False
):
    """Creates a PyMAS wrapper to execute a function defined in an
    external .py file.

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
        Whether the function inputs should be treated as an array instead of
        individual parameters

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
        raise RuntimeError("Could not find a valid function named %s" % func_name)

    with open(file, 'r') as f:
        code = [line.strip('\n') for line in f.readlines()]

    return _build_pymas(target_func, None, input_types, array_input, code=code)


@versionadded(version='1.6')
def from_model_info(info):
    """Create a deployable DS2 package from a ModelInfo instance.

    Parameters
    ----------
    info : ModelInfo

    Returns
    -------
    PyMAS
        Generated DS2 code which can be executed in a SAS scoring environment

    """

    # Encode the pickled data so we can inline it in the DS2 package
    pkl = base64.b64encode(pickle.dumps(info.instance))

    code = (
        'import pickle, base64',
        # Replace b' with " before embedding in DS2.
        'bytes = {}'.format(pkl).replace("'", '"'),
        'obj = pickle.loads(base64.b64decode(bytes))',
    )

    for name in info.function_names:
        if name not in info.input_variables:
            raise ValueError("Input variable information missing for function '%s'." % name)
        if name not in info.output_variables:
            raise ValueError("Output variable information missing for function '%s'." % name)

    # No need to use _build_pymas() - already have variable information by function.
    # Just need to convert to DS2Variable instances and combine into 1 list per function
    variables = [ds2_variables(info.input_variables[f]) + ds2_variables(info.output_variables[f], output_vars=True) for f in info.function_names]

    return PyMAS2(info)


def from_pickle(
    file,
    func_name=None,
    input_types=None,
    array_input=False
):
    """Create a deployable DS2 package from a Python pickle file.

    Parameters
    ----------
    file : str or bytes or file_like
        Pickled object to use.  String is assumed to be a path to a picked
        file, file_like is assumed to be an open file handle to a pickle
        object, and bytes is assumed to be the raw pickled bytes.
    func_name : str
        Name of the target function to call
    input_types : DataFrame, type, list of type, or dict of str: type, optional
        The expected type for each input value of the target function.
        Can be omitted if target function includes type hints.  If a DataFrame
        is provided, the columns will be inspected to determine type information.
        If a single type is provided, all columns will be assumed to be that type,
        otherwise a list of column types or a dictionary of column_name: type
        may be provided.
    array_input : bool
        Whether the function inputs should be treated as an array instead of
        individual parameters

    Returns
    -------
    PyMAS
        Generated DS2 code which can be executed in a SAS scoring environment

    """
    try:
        # In Python2 str could either be a path or the binary pickle data,
        # so check if its a valid filepath too.
        is_file_path = isinstance(file, str) and os.path.isfile(file)
    except TypeError:
        is_file_path = False

    # Path to a pickle file
    if is_file_path:
        with open(file, 'rb') as f:
            obj = pickle.load(f)  # skipcq BAN-B301

    # The actual pickled bytes
    elif isinstance(file, bytes):
        obj = pickle.loads(file)  # skipcq BAN-B301
    else:
        obj = pickle.load(file)  # skipcq BAN-B301

    # Encode the pickled data so we can inline it in the DS2 package
    pkl = base64.b64encode(pickle.dumps(obj))

    code = (
        'import pickle, base64',
        # Replace b' with " before embedding in DS2.
        'bytes = {}'.format(pkl).replace("'", '"'),
        'obj = pickle.loads(base64.b64decode(bytes))',
    )

    return _build_pymas(
        obj, func_name, input_types, array_input, func_prefix='obj.', code=code
    )





def _build_pymas(
    obj,
    func_name=None,
    input_types=None,
    array_input=False,
    func_prefix=None,
    code=None,
):
    """

    Parameters
    ----------
    obj
    func_name
    input_types
    array_input
    return_code
    return_message
    code

    Returns
    -------
    PyMAS

    """
    code = code or []

    if not isinstance(func_name, list):
        func_name = [func_name]

    def parse_function(obj, func_name):
        # If the object passed was a function, no need to search for
        # target function
        if callable(obj) and (func_name is None or obj.__name__ == func_name):
            target_func = obj
        elif func_name is None:
            raise ValueError('Parameter `func_name` must be specified.')
        else:
            target_func = getattr(obj, func_name)

        if not callable(target_func):
            raise RuntimeError("Could not find a valid function named %s" % func_name)

        target_func_name = target_func.__name__

        if target_func_name.lower() == 'predict_proba':
            names = 'P_'
        else:
            names = None

        # Need to create DS2Variable instances to pass to PyMAS
        if hasattr(input_types, 'columns'):
            # Assuming input is a DataFrame representing model inputs.  Use to
            # get input variables
            vars = ds2_variables(input_types)

            # Run one observation through the model and use the result to
            # determine output variables
            output = target_func(input_types.head(1))
            output_vars = ds2_variables(output, output_vars=True, names=names)
            vars.extend(output_vars)
        elif isinstance(input_types, type):
            params = OrderedDict(
                [(k, input_types) for k in target_func.__code__.co_varnames]
            )
            vars = ds2_variables(params)
        elif isinstance(input_types, dict):
            vars = ds2_variables(input_types)
        else:
            # Inspect the Python method to determine arguments
            vars = ds2_variables(target_func)

        if not any(v for v in vars if v.out):
            vars.append(DS2Variable(name='result', type='float', out=True))

        return target_func_name, vars

    # Get variables for each function
    temp = [parse_function(obj, f) for f in func_name]
    target_func, variables = zip(*temp)

    # Convert from tuples to lists
    target_func = list(target_func)
    variables = list(variables)

    return PyMAS(
        target_func, variables, code, array_input=array_input, func_prefix=func_prefix
    )


class PyMAS:
    """

    Parameters
    ----------
    target_function : str or  list
        The Python function to be executed.
    variables : list of DS2Variable  or list of list
        The input/ouput variables be declared in the module.
    python_source : str
        Additional Python code to be executed during setup.
    kwargs : any
        Passed to :func:`build_wrapper_function`.

    Attributes
    ----------
    wrapper : str
        Python code containing generated methods to execute model methods.


    """

    def __init__(
        self,
        target_function,
        variables,
        python_source,
        func_prefix=None,
        **kwargs
    ):

        func_prefix = func_prefix or ''

        if not isinstance(target_function, list):
            target_function = [target_function]
            variables = [variables]

        return_code = False
        return_msg = False

        # Replicate parameter for each function to be exposed
        if isinstance(return_code, bool):
            return_code = [return_code] * len(target_function)

        # Replicate parameter for each function to be exposed
        if isinstance(return_msg, bool):
            return_msg = [return_msg] * len(target_function)

        self.wrapper = []
        wrapper_names = []
        for func, func_vars, code, msg in zip(
            target_function, variables, return_code, return_msg
        ):
            # Random name to avoid conflict with any existing methods
            wrapper_names.append('_' + random_string(20))

            if func.lower() == 'predict':
                lines = wrap_predict_method(
                    func_prefix + func,
                    func_vars,
                    setup=python_source,
                    name=wrapper_names[-1],
                    **kwargs
                )
            elif func.lower() == 'predict_proba':
                lines = wrap_predict_proba_method(
                    func_prefix + func,
                    func_vars,
                    name=wrapper_names[-1],
                    setup=python_source,
                    **kwargs
                )
            else:
                lines = build_wrapper_function(
                    func_prefix + func,
                    func_vars,
                    name=wrapper_names[-1],
                    setup=python_source,
                    **kwargs
                )

            # # Add DS2 variables for returning error codes/messages
            # # NOTE: add these *after* wrapper function is generated to prevent
            # # double-counting them.
            # if code:
            #     func_vars.append(DS2Variable(name='rc', type='int32', out=True))
            # if msg:
            #     func_vars.append(DS2Variable(name='msg', type='char', out=True))

            # Clear setup code once it's been added once.  No need to duplicate
            # if multiple functions are defined.
            python_source = None

            self.wrapper.extend(lines.split('\n'))

        self.variables = variables[0]
        # self.return_code = return_code[0]
        self.return_message = return_msg[0]



        self.package = DS2PyMASPackage(self.wrapper)

        for idx, func in enumerate(target_function):
            self.package.add_method(func, wrapper_names[idx], variables[idx])

    @versionchanged(version='1.4', reason="Added `dest='Python'` option")
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
        dest : str {'MAS', 'EP', 'CAS', 'Python'}
            Specifies the publishing destination for the score code to ensure
            that compatible code is generated.

        Returns
        -------
        str
            Score code

        """

        dest = dest.upper()

        # Check for names that could result in DS2 errors.
        DS2_KEYWORDS = ['input', 'output']
        for k in DS2_KEYWORDS:
            if input_table and k == input_table.lower():
                raise ValueError(
                    'Input table name `{}` is a reserved term.'.format(input_table)
                )
            if output_table and k == output_table.lower():
                raise ValueError(
                    'Output table name `{}` is a reserved term.'.format(output_table)
                )

        # Get package code
        code = tuple(self.package.code().split('\n'))

        if dest == 'EP':
            code = (
                ('data sasep.out;',)
                + code
                + (
                    '   method run();',
                    '      set SASEP.IN;',
                    '   end;',
                    '   method term();',
                    '   end;',
                    'enddata;',
                )
            )
        elif dest == 'CAS':
            thread = DS2Thread(
                self.variables,
                input_table,
                column_names=columns,
                return_message=self.return_message,
                package=self.package,
            )

            code += (
                str(thread),
                'data SASEP.out;',
                '  dcl thread {} t;'.format(thread.name),
                '  method run();',
                '    set from t;',
                '    output;',
                '  end;',
                'enddata;',
            )

        elif dest == 'PYTHON':
            # Python code return
            code = self.package._python_code

        return '\n'.join(code)


class PyMAS2:
    def __init__(self, model_info):

        # init code
        # score resources

        # Name of DS2 method that is used by default
        default_method_name = None

        # assume model_init_code() preps an object called 'model'
        function_prefix = 'model.'
        lines = []
        ds2_methods = []

        # Get any init code needed to load/ready the model
        init_code = self.model_init_code(model_info)

        # Wrap in a try/catch block so we can return errors if necessary
        init_code = ('try:',) + tuple('    ' + line for line in init_code) + ('    _compile_error = None',
                                                                              'except Exception as e:',
                                                                              '    _compile_error = e',
                                                                              '',
                                                                              )
        lines.extend(init_code)

        # Generate wrappers
        for name in model_info.function_names:
            # Random name to avoid conflict with any existing methods
            wrapper_name = '_' + random_string(20)

            variables = ds2_variables(model_info.input_variables[name]) + ds2_variables(
                model_info.output_variables[name], output_vars=True)

            if name.lower() == 'predict':
                # Default to .predict if found
                default_method_name = name

                code = wrap_predict_method(function_prefix + name,
                                           variables,
                                           name=wrapper_name
                                           )
            elif name.lower() == 'predict_proba':
                code = wrap_predict_proba_method(function_prefix + name,
                                                 variables,
                                                 name=wrapper_name
                                                 )
            else:
                # Use current mystery method only if we haven't found something more standard
                default_method_name = default_method_name or name.lower()

                code = build_wrapper_function(function_prefix + name,
                                              variables,
                                              model_info.array_input,
                                              name=wrapper_name
                                              )
                # Track mapping of Python wrapper method => DS2 method
            # Will be needed later to generate corresponding DS2 methods
            ds2_methods.append({
                'name': name,
                'target': wrapper_name,
                'variables': variables
            })

            # Append code for new wrapper method to list of generated Python code
            lines.extend(code.split('\n'))

        # Generate a "score" method for classification models
        if model_info.is_classification and 'predict_proba' in model_info.function_names:
            name = 'predict_proba'
            wrapper_name = '_' + random_string(20)
            variables = ds2_variables(model_info.input_variables[name]) + ds2_variables(
                model_info.output_variables[name], output_vars=True)

            code = wrap_classification_score_method(function_prefix + name,
                                                    variables,
                                                    model_info.class_names,
                                                    model_info.target_level,
                                                    name=wrapper_name)
            ds2_methods.append({
                'name': 'score',
                'target': wrapper_name,
                'variables': variables
            })
            lines.extend(code.split('\n'))

            # Use score method if created
            default_method_name = 'score'

        elif model_info.is_regression and 'predict' in model_info.function_names:
            name = 'predict'
            wrapper_name = '_' + random_string(20)
            variables = ds2_variables(model_info.input_variables[name]) + ds2_variables(
                model_info.output_variables[name], output_vars=True)

            code = wrap_regression_score_method(function_prefix + name,
                                                variables,
                                                name=wrapper_name)
            ds2_methods.append({
                'name': 'score',
                'target': wrapper_name,
                'variables': variables
            })
            lines.extend(code.split('\n'))

            # Use score method if created
            default_method_name = 'score'

        # Create a DS2 package that automatically loads all of the generated Python code into a PyMAS instance.
        self.package = DS2PyMASPackage(lines)

        # Just for debugging / inspection
        self.lines = lines

        self._model_info = model_info

        # Add DS2 methods for each of the wrapped Python functions
        for method in ds2_methods:
            self.package.add_method(**method)

        self.default_method = next((m for m in self.package.methods if m.name == default_method_name), None)

    @classmethod
    def model_init_code(cls, model_info):
        # attempt to pickle & inline
        # fall back to file
        if model_info.tool == 'pytorch':
            # TODO: import torch
            # torch.save(temp file name)
            # return code to load
            # !! needs model id - hasn't been created yet.  F.
            pass
        else:
            pkl = base64.b64encode(gzip.compress(pickle.dumps(model_info.instance)))

            code = (
                'import base64, gzip, pickle',
                # Replace b' with " before embedding in DS2.
                'bytes = {}'.format(pkl).replace("'", '"'),
                'model = pickle.loads(gzip.decompress(base64.b64decode(bytes)))',
            )

        return code

    @versionchanged(version='1.4', reason="Added `dest='Python'` option")
    def score_code(self, input_table=None, output_table=None, columns=None, dest='MAS', method_name=None):
        """Generate DS2 score code

        Parameters
        ----------
        input_table : str
            The name of the table to execute the function against
        output_table : str
            The name of the table where execution results will be written
        columns : list of str
            Names of the columns from `table` that will be passed to `func`
        dest : str {'MAS', 'EP', 'CAS', 'Python'}
            Specifies the publishing destination for the score code to ensure
            that compatible code is generated.

        Returns
        -------
        str
            Score code

        """

        dest = dest.upper()

        # Check for names that could result in DS2 errors.
        DS2_KEYWORDS = ['input', 'output']
        for k in DS2_KEYWORDS:
            if input_table and k == input_table.lower():
                raise ValueError(
                    'Input table name `{}` is a reserved term.'.format(input_table)
                )
            if output_table and k == output_table.lower():
                raise ValueError(
                    'Output table name `{}` is a reserved term.'.format(output_table)
                )

        # Get package code
        code = tuple(self.package.code().split('\n'))

        if dest == 'EP':
            code = (
                ('data sasep.out;',)
                + code
                + (
                    '   method run();',
                    '      set SASEP.IN;',
                    '   end;',
                    '   method term();',
                    '   end;',
                    'enddata;',
                )
            )
        elif dest == 'CAS':
            if method_name is None:
                method = self.default_method
            else:
                method = next((m for m in self.package.methods if m.name == method_name), None)

            thread = DS2Thread(
                method.variables,
                input_table,
                column_names=columns,
                return_message=False,
                method=method,
                package=self.package,
            )

            code += (
                str(thread),
                'data SASEP.out;',
                '  dcl thread {} t;'.format(thread.name),
                '  method run();',
                '    set from t;',
                '    output;',
                '  end;',
                'enddata;',
            )

        elif dest == 'PYTHON':
            # Python code return
            code = self.package._python_code

        return '\n'.join(code)
