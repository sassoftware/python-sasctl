#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from collections import OrderedDict
from inspect import getfullargspec, getsourcelines

from .ds2 import DS2Variable
from ..decorators import versionchanged


logger = logging.getLogger(__name__)


@versionchanged("`names` parameter added.", version="1.5")
def ds2_variables(input, output_vars=False, names=None):
    """Generate a collection of `DS2Variable` instances corresponding to the input

    Parameters
    ----------
    input : function or OrderedDict<string, tuple> or Pandas DataFrame or Numpy or OrderedDict<string, type>
        a function or mapping parameter names to (type, is_output)

    output_vars : bool
        Whether or not to treat  all variables from `input` as output variables

    names : list of str
        List of variable names to used.  If a single string is specified it will
        be used as a prefix and variable names in the format "prefixNN" will be
        generated.

    Returns
    -------
    list<DS2Variable>

    Examples
    --------
    >>> ds2_variables(OrderedDict(a=int, c=float))
    [DS2Variable(name='a', type='integer', out=False), DS2Variable(name='c', type='double', out=False)]

    >>> ds2_variables({'x': (float, True)})
    [DS2Variable(name='x', type='double', out=True)]

    """

    if isinstance(input, dict):
        types = input
    elif hasattr(input, "columns") and hasattr(input, "dtypes"):
        # Pandas DataFrame
        types = OrderedDict()
        for col in input.columns:
            if input[col].dtype.name == "object":
                types[col] = ("char", False)
            elif input[col].dtype.name == "category":
                types[col] = ("char", False)
            else:
                types[col] = (input[col].dtype.name, False)
    elif hasattr(input, "dtype"):
        # Numpy array?  No column names, but we can at least create dummy vars of the correct type
        types = OrderedDict(
            [
                ("var{}".format(i), (input.dtype.name.replace("object", "char"), False))
                for i in range(1, input.size + 1)
            ]
        )
    elif callable(input):
        types = parse_type_hints(input)
    else:
        raise RuntimeError(
            "Unable to determine input/ouput types using "
            "instance of type '%s'." % type(input)
        )

    if isinstance(names, str):
        names = [names + str(i) for i in range(1, len(types) + 1)]
    elif names is None:
        names = list(types.keys())

    results = []
    for v in types.values():
        name = names.pop(0)
        if isinstance(v, str):
            results.append(DS2Variable(name=name, type=v, out=output_vars))
        elif isinstance(v, type):
            results.append(DS2Variable(name=name, type=v.__name__, out=output_vars))
        elif isinstance(v, tuple):
            type_ = v[0].__name__ if isinstance(v[0], type) else str(v[0])
            out = v[1] or output_vars
            results.append(DS2Variable(name=name, type=type_, out=out))
        else:
            raise RuntimeError("Unable to determine input/ouput types.")

    return results


def parse_type_hints(func, skip_var="self"):
    """Attempt to discern types for the input and output variable(s).

    DS2 is a strongly-typed language but Python is not.  Need to determine the types for input/output variables so they
    can be correctly mapped to DS2 types.

    Parameters
    ----------
    func : function
        the object to inspect for parameters

    skip_var : str
        name of the variable assumed to be the instance reference, if any.  Will be ignored during parameter parsing.

    Returns
    -------
    params : OrderedDict
        dictionary of str : (str, bool) mapping param_name : (param_type, is_return_val)

    Raises
    ------
    ValueError
        If the type of one or more variables could not be determined.

    """

    params = OrderedDict(
        [(k, None) for k in getfullargspec(func).args if k != skip_var]
    )
    logger.debug("Params: {}".format(params))

    if getattr(func, "__annotations__", None):
        params.update(parse_type_hints_from_annotations(func, skip_var=skip_var))
    else:
        params.update(parse_type_hints_from_source(func, skip_var=skip_var))

    if any(v is None for v in params.values()):
        raise ValueError("Unable to determine parameter types.")

    return params


def parse_type_hints_from_source(func, skip_var="self"):
    """Parse type hints stored in comments according to PEP 484."""

    regex = re.compile(r"^\s+\# types?: ", re.IGNORECASE)

    def parse_types(line):
        if line:
            line = regex.sub("", line)  # Strip out the ' # type:' portion if it exists
            return line.strip().strip("(").strip(")").split(",")
        return []

    params = OrderedDict()

    for line in getsourcelines(func)[0]:
        if regex.match(line):
            if "->" in line:
                inputs, outputs = line.split("->")
            else:
                inputs = line
                outputs = None

            types = parse_types(inputs)
            args = getfullargspec(func).args  # skipcq PYL-W1505

            try:
                args.remove(skip_var)
            except ValueError:
                pass

            for a, t in zip(args, types):
                params[a] = (t.strip(), False)

            types = parse_types(outputs)
            for i, t in enumerate(types):
                params["out%d" % (i + 1)] = (t.strip(), True)

    return params


def parse_type_hints_from_annotations(func, skip_var="self"):
    """Parse type hints from the function signature."""

    annotations = func.__annotations__
    params = OrderedDict()
    logger.debug("Annotations: {}".format(annotations))

    for p, t in annotations.items():
        if p == skip_var:
            continue
        elif p == "return":
            if t is not None:
                params[p] = (t.__name__, True)
        else:
            params[p] = (t.__name__, False)

    return params
