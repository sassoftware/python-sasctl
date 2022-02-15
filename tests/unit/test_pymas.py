#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest
from unittest import mock

from sasctl.utils.pymas.core import build_wrapper_function, PyMAS
from sasctl.utils.pymas.ds2 import DS2Variable
from sasctl.utils.pymas.python import ds2_variables, parse_type_hints


class DummyClass:
    """Used for pickling tests."""

    def func(self, x1, x2):
        # type: (str, int) -> (float)
        pass


def dummy_func(x1, x2):
    # type: (str, int) -> (float)
    pass


def test_score_code():
    from sasctl.utils.pymas import from_inline

    p = from_inline(dummy_func)
    assert isinstance(p, PyMAS)

    mas_code = p.score_code()
    esp_code = p.score_code(dest='ep')

    assert mas_code.lower().startswith('package _')
    assert esp_code.lower().startswith('data sasep.out;')

    cas_code = p.score_code('in_table', 'out_table', ['in1', 'in2'], dest='cas')

    pytest.xfail('Not implemented.')


def test_from_inline():
    """Create a PyMAS instance from an inline Python function."""
    from sasctl.utils.pymas import from_inline

    result = from_inline(dummy_func)
    assert isinstance(result, PyMAS)

    with mock.patch('sasctl.utils.pymas.core.PyMAS', autospec=True) as mocked:
        _ = from_inline(dummy_func)
        assert 1 == mocked.call_count
        call_args = mocked.call_args[0]
        assert [
            [
                DS2Variable('x1', 'str', False),
                DS2Variable('x2', 'int', False),
                DS2Variable('out1', 'float', True),
            ]
        ] == call_args[
            1
        ]  # Variables

    with mock.patch('sasctl.utils.pymas.core.PyMAS', autospec=True) as mocked:
        _ = from_inline(dummy_func, input_types=int)
        assert 1 == mocked.call_count
        call_args = mocked.call_args[0]
        assert [
            [
                DS2Variable('x1', 'int', False),
                DS2Variable('x2', 'int', False),
                DS2Variable('result', 'float', True),
            ]
        ] == call_args[
            1
        ]  # Variables

    with mock.patch('sasctl.utils.pymas.core.PyMAS', autospec=True) as mocked:
        _ = from_inline(dummy_func, input_types=OrderedDict([('a', int), ('b', float)]))
        assert 1 == mocked.call_count
        call_args = mocked.call_args[0]
        assert [
            [
                DS2Variable('a', 'int', False),
                DS2Variable('b', 'double', False),
                DS2Variable('result', 'float', True),
            ]
        ] == call_args[
            1
        ]  # Variables


def test_from_pickle_with_func():
    """Create a PyMAS instance from a pickled object."""

    import pickle
    from sasctl.utils.pymas import from_pickle

    data = pickle.dumps(dummy_func)

    with mock.patch('sasctl.utils.pymas.core.PyMAS', autospec=True) as mocked:
        result = from_pickle(data)
        assert 1 == mocked.call_count
        call_args = mocked.call_args[0]
        assert [
            [
                DS2Variable('x1', 'str', False),
                DS2Variable('x2', 'int', False),
                DS2Variable('out1', 'float', True),
            ]
        ] == call_args[
            1
        ]  # Variables

    assert isinstance(result, PyMAS)


def test_from_pickle_with_class():
    """Create a PyMAS instance from a pickled object."""

    import pickle
    from sasctl.utils.pymas import from_pickle

    data = pickle.dumps(DummyClass())

    with pytest.raises(ValueError):
        result = from_pickle(data)  # No function specified

    with mock.patch('sasctl.utils.pymas.core.PyMAS', autospec=True) as mocked:
        result = from_pickle(data, 'func')
        assert 1 == mocked.call_count
        call_args = mocked.call_args[0]
        assert [
            [
                DS2Variable('x1', 'str', False),
                DS2Variable('x2', 'int', False),
                DS2Variable('out1', 'float', True),
            ]
        ] == call_args[
            1
        ]  # Variables

    assert isinstance(result, PyMAS)


def test_build_wrapper_function():
    def func(a, b):
        return a, b

    # Actual function inputs & DS2 variables current dont have to match.
    result = build_wrapper_function(
        func,
        [DS2Variable('a', 'int', False), DS2Variable('x', 'float', True)],
        array_input=False,
    )
    assert isinstance(result, str)
    assert '"Output: x' in result


def test_ds2_double():
    # Input variable
    var = DS2Variable('myvar', 'float', False)
    assert 'double' == var.type
    assert False == var.is_array
    assert 'dcl double myvar;' == var.as_declaration()
    assert 'double myvar' == var.as_parameter()
    assert "rc = py.setDouble('myvar', myvar);" == var.pymas_statement()

    # Output variable
    var = DS2Variable('myvar', 'double', True)
    assert 'double' == var.type
    assert False == var.is_array
    assert 'dcl double myvar;' == var.as_declaration()
    assert 'in_out double myvar' == var.as_parameter()
    assert "myvar = py.getDouble('myvar');" == var.pymas_statement()


def test_ds2_char():
    # Input variable
    var = DS2Variable('myvar', 'str', False)
    assert 'char' == var.type
    assert False == var.is_array
    assert 'dcl char myvar;' == var.as_declaration()
    assert 'char myvar' == var.as_parameter()
    assert "rc = py.setString('myvar', myvar);" == var.pymas_statement()

    # Output variable
    var = DS2Variable('myvar', 'string', True)
    assert 'char' == var.type
    assert False == var.is_array
    assert 'dcl char myvar;' == var.as_declaration()
    assert 'in_out char myvar' == var.as_parameter()
    assert "myvar = py.getString('myvar');" == var.pymas_statement()


def test_ds2_int():
    # Input variable
    var = DS2Variable('myvar', 'int', False)
    assert 'integer' == var.type
    assert False == var.is_array
    assert 'dcl integer myvar;' == var.as_declaration()
    assert 'integer myvar' == var.as_parameter()
    assert "rc = py.setInt('myvar', myvar);" == var.pymas_statement()

    # Output variable
    var = DS2Variable('myvar', 'integer', True)
    assert 'integer' == var.type
    assert False == var.is_array
    assert 'dcl integer myvar;' == var.as_declaration()
    assert 'in_out integer myvar' == var.as_parameter()
    assert "myvar = py.getInt('myvar');" == var.pymas_statement()


def test_ds2_variables_dict_input():
    target = [
        DS2Variable(name='x1', type='double', out=False),
        DS2Variable(name='x2', type='double', out=False),
    ]

    # Standard input
    result = ds2_variables(
        OrderedDict([('x1', ('double', False)), ('x2', ('double', False))])
    )
    assert result == target

    # No return value flag
    result = ds2_variables(OrderedDict([('x1', 'double'), ('x2', 'double')]))
    assert result == target

    assert [
        DS2Variable('a', 'int', False),
        DS2Variable('b', 'char', False),
        DS2Variable('c', 'double', False),
    ] == ds2_variables(OrderedDict([('a', int), ('b', str), ('c', float)]))

    assert [DS2Variable('x', 'str', True)] == ds2_variables(
        OrderedDict(x=str), output_vars=True
    )

    assert [
        DS2Variable('a', 'int', False),
        DS2Variable('b', 'char', False),
        DS2Variable('x', 'double', True),
    ] == ds2_variables(OrderedDict([('a', int), ('b', str), ('x', (float, True))]))

    assert [
        DS2Variable('a', 'uint8', False),
        DS2Variable('b', 'uint16', False),
        DS2Variable('c', 'uint32', False),
        DS2Variable('d', 'uint64', False),
    ] == ds2_variables(OrderedDict([('a', int), ('b', int), ('c', int), ('d', int)]))


def test_ds2_variables_func_input():
    target = [
        DS2Variable(name='x1', type='double', out=False),
        DS2Variable(name='x2', type='double', out=False),
    ]

    # Only function as input
    with mock.patch('sasctl.utils.pymas.python.parse_type_hints') as mocked:
        mocked.return_value = OrderedDict(
            [('x1', ('double', False)), ('x2', ('double', False))]
        )
        result = ds2_variables(
            lambda x: x
        )  # Input function doesn't matter since response is mocked
        assert 1 == mocked.call_count
    assert result == target


def test_ds2_variables_pandas_input():
    pd = pytest.importorskip('pandas')

    target = [
        DS2Variable('a', 'int', False),
        DS2Variable('b', 'char', False),
        DS2Variable('c', 'double', False),
    ]

    df = pd.DataFrame(dict(a=[1, 2, 3], b=['1', '2', '3'], c=[1.0, 2.0, 3.0]))
    df = df[['a', 'b', 'c']]  # Ensure columns are ordered correctly

    assert target == ds2_variables(df)


def test_ds2_variables_numpy_input():
    np = pytest.importorskip('numpy')

    assert [
        DS2Variable('var1', 'int', False),
        DS2Variable('var2', 'int', False),
        DS2Variable('var3', 'int', False),
    ] == ds2_variables(np.array([1, 2, 3]))

    assert [
        DS2Variable('var1', 'double', False),
        DS2Variable('var2', 'double', False),
    ] == ds2_variables(np.array([1.0, 2.0]))


def test_parse_type_hints_source():
    def bad_func(x1, x2):
        return x1 + x2

    with pytest.raises(ValueError):
        parse_type_hints(bad_func)

    def func1(x1, x2):
        # type: (str, int) -> (float, bool)
        pass

    result = parse_type_hints(func1)
    assert result == OrderedDict(
        [
            ('x1', ('str', False)),
            ('x2', ('int', False)),
            ('out1', ('float', True)),
            ('out2', ('bool', True)),
        ]
    )

    def func2(x1, x2):
        # type: float, float -> float
        pass

    result = parse_type_hints(func2)
    assert result == OrderedDict(
        [('x1', ('float', False)), ('x2', ('float', False)), ('out1', ('float', True))]
    )

    def func3(x1, x2):
        # type: float, float -> float, int
        pass

    result = parse_type_hints(func3)
    assert result == OrderedDict(
        [
            ('x1', ('float', False)),
            ('x2', ('float', False)),
            ('out1', ('float', True)),
            ('out2', ('int', True)),
        ]
    )

    def func4(x1, x2):
        # type: str, int
        pass

    result = parse_type_hints(func4)
    assert result == OrderedDict([('x1', ('str', False)), ('x2', ('int', False))])

    def func5(x1, x2):
        # type: (str, int)
        pass

    result = parse_type_hints(func5)
    assert result == OrderedDict([('x1', ('str', False)), ('x2', ('int', False))])

    def func6(self, x1, x2):
        # type: (str, int)
        pass

    result = parse_type_hints(func6)
    assert result == OrderedDict([('x1', ('str', False)), ('x2', ('int', False))])

    class TestClass:
        def func(self, x1, x2):
            # type: (str, int)
            pass

    result = parse_type_hints(TestClass().func)
    assert result == OrderedDict([('x1', ('str', False)), ('x2', ('int', False))])


def test_type_hint_annotations():
    # Annotated functions in the form `def annotated_func(x1: int, x2: int):` will cause syntax errors in Python 2
    # To enable testing on all supported runtimes, fake the annotations by setting __annotations__ directly
    def annotated_func(x1, x2):
        pass

    annotated_func.__annotations__ = {'x1': int, 'x2': int}
    result = parse_type_hints(annotated_func)
    assert result == dict(x1=('int', False), x2=('int', False))


def test_ds2_pymas_method():
    """Test DS2 code generation for a method that uses PyMAS."""
    from sasctl.utils.pymas.ds2 import DS2PyMASMethod, DS2Variable

    source = """
def domath(a, b):
    "Output: c, d"
    c = a * b
    d = a / b
    return c, d
"""

    target = """
method score(
    double a,
    double b,
    in_out double c,
    in_out double d,
    in_out integer rc
    );

    if null(py) then do;
        py = _new_ pymas();
        rc = py.useModule('mypymodule', 1);
        if rc then do;
            rc = py.appendSrcLine('def domath(a, b):');
            rc = py.appendSrcLine('    "Output: c, d"');
            rc = py.appendSrcLine('    c = a * b');
            rc = py.appendSrcLine('    d = a / b');
            rc = py.appendSrcLine('    return c, d');
            pycode = py.getSource();
            revision = py.publish(pycode, 'mypymodule');
            if revision lt 1 then do;
                logr.log('e', 'py.publish() failed.');
                rc = -1;
            end;
        end;
        rc = py.useMethod('domath');
        if rc then return;
    end;
    rc = py.setDouble('a', a);    if rc then return;
    rc = py.setDouble('b', b);    if rc then return;
    rc = py.execute();    if rc then return;
    c = py.getDouble('c');
    d = py.getDouble('d');
end;
"""
    method = DS2PyMASMethod(
        'mypymodule',
        [
            DS2Variable('a', 'double', False),
            DS2Variable('b', 'double', False),
            DS2Variable('c', 'double', True),
            DS2Variable('d', 'double', True),
        ],
        source.strip('\n'),
        return_code=True,
        return_message=False,
        target='domath',
    )

    result = method.code()
    assert target.lstrip('\n') == result
    assert '\t' not in result


def test_ds2_base_method():
    """Test DS2 code generation for a simple DS2 method."""
    from sasctl.utils.pymas.ds2 import DS2BaseMethod, DS2Variable

    # Generate a empty method
    target = """
method testmethod(
    double a,
    double b,
    in_out double c,
    in_out double d
    );

end;
"""
    method = DS2BaseMethod(
        'testmethod',
        [
            DS2Variable('a', 'double', False),
            DS2Variable('b', 'double', False),
            DS2Variable('c', 'double', True),
            DS2Variable('d', 'double', True),
        ],
    )

    result = method.code()
    assert target.lstrip('\n') == result
    assert '\t' not in result

    # Generate a method with an arbitrary body
    target = """
method testmethod(
    double a,
    double b,
    in_out double c,
    in_out double d
    );

    if parrot.is_dead:
        return
end;
"""
    method = DS2BaseMethod(
        'testmethod',
        [
            DS2Variable('a', 'double', False),
            DS2Variable('b', 'double', False),
            DS2Variable('c', 'double', True),
            DS2Variable('d', 'double', True),
        ],
        ['if parrot.is_dead:', '    return'],
    )

    result = method.code()
    assert target.lstrip('\n') == result
    assert '\t' not in result


def test_ds2_package():
    """Test code generation for a package with PyMAS."""
    from sasctl.utils.pymas.ds2 import DS2Package, DS2Variable

    target = """
package _pyscore / overwrite=yes;
    dcl package pymas py;
    dcl package logger logr('App.tk.MAS');
    dcl varchar(67108864) character set utf8 pycode;
    dcl int revision;

    method score(
        double a,
        double b,
        in_out double c,
        in_out double d,
        in_out integer rc
        );
    
        if null(py) then do;
            py = _new_ pymas();
            rc = py.useModule('pyscore', 1);
            if rc then do;
                rc = py.appendSrcLine('def domath(a, b):');
                rc = py.appendSrcLine('    "Output: c, d"');
                rc = py.appendSrcLine('    c = a * b');
                rc = py.appendSrcLine('    d = a / b');
                rc = py.appendSrcLine('    return c, d');
                pycode = py.getSource();
                revision = py.publish(pycode, 'pyscore');
                if revision lt 1 then do;
                    logr.log('e', 'py.publish() failed.');
                    rc = -1;
                end;
            end;
            rc = py.useMethod('domath');
            if rc then return;
        end;
        rc = py.setDouble('a', a);    if rc then return;
        rc = py.setDouble('b', b);    if rc then return;
        rc = py.execute();    if rc then return;
        c = py.getDouble('c');
        d = py.getDouble('d');
    end;
    
endpackage;
"""

    source = """
def domath(a, b):
    "Output: c, d"
    c = a * b
    d = a / b
    return c, d
"""

    with mock.patch('uuid.uuid4') as mocked:
        mocked.return_value.hex.upper.return_value = 'pyscore'
        package = DS2Package(
            [
                DS2Variable('a', 'double', False),
                DS2Variable('b', 'double', False),
                DS2Variable('c', 'double', True),
                DS2Variable('d', 'double', True),
            ],
            source.strip('\n'),
            target='domath',
            return_message=False,
        )

    result = package.code()
    assert target.lstrip('\n') == result
    assert '\t' not in result


def test_bugfix_27():
    """NaN values should be converted to null before being sent to MAS

    https://github.com/sassoftware/python-sasctl/issues/27
    """

    import io
    from sasctl.core import RestObj
    from sasctl.services import microanalytic_score as mas

    pd = pytest.importorskip('pandas')

    df = pd.read_csv(
        io.StringIO(
            '\n'.join(
                [
                    u'BAD,LOAN,MORTDUE,VALUE,REASON,JOB,YOJ,DEROG,DELINQ,CLAGE,NINQ,CLNO,DEBTINC',
                    u'0,1.0,1100.0,25860.0,39025.0,HomeImp,Other,10.5,0.0,0.0,94.36666667,1.0,9.0,',
                    u'1,1.0,1300.0,70053.0,68400.0,HomeImp,Other,7.0,0.0,2.0,121.8333333,0.0,14.0,',
                ]
            )
        )
    )

    with mock.patch(
        'sasctl._services.microanalytic_score.MicroAnalyticScore.get_module'
    ) as get_module:
        get_module.return_value = RestObj({'name': 'Mock Module', 'id': 'mockmodule'})
        with mock.patch(
            'sasctl._services.microanalytic_score.MicroAnalyticScore.post'
        ) as post:
            x = df.iloc[0, :]
            mas.execute_module_step('module', 'step', **x)

    # Ensure we're passing NaN to execute_module_step
    assert pd.isna(x['DEBTINC'])

    # Make sure the value has been converted to None before being serialized to JSON.
    # This ensures that the JSON value will be null.
    json = post.call_args[1]['json']
    inputs = json['inputs']
    debtinc = [i for i in inputs if i['name'] == 'DEBTINC'].pop()
    assert debtinc['value'] is None


def test_wrapper():
    """Verify correct output from build_wrapper_function under default settings."""

    target = """
def wrapper(a, b):
    "Output: c, msg"
    result = None
    msg = None
    try:
        global _compile_error
        if _compile_error is not None:
            raise _compile_error
        import numpy as np
        import pandas as pd

        if a is None: a = np.nan
        if b is None: b = np.nan
        input_array = np.array([a, b]).reshape((1, -1))
        columns = ["a", "b"]
        input_df = pd.DataFrame(data=input_array, columns=columns)
        result = dummy_func(input_df)
        result = tuple(result.ravel()) if hasattr(result, "ravel") else tuple(result)
        if len(result) == 0:
            result = tuple(None for i in range(1))
        elif "numpy" in str(type(result[0])):
            result = tuple(np.asscalar(i) for i in result)
    except Exception as e:
        from traceback import format_exc
        msg = str(e) + format_exc()
        if result is None:
            result = tuple(None for i in range(1))
    return result + (msg, )
    """.rstrip()

    code = build_wrapper_function(
        dummy_func,
        [
            DS2Variable('a', float, False),
            DS2Variable('b', float, False),
            DS2Variable('c', float, True),
        ],
        array_input=True,
    )

    assert code == target


def test_wrapper_renamed():
    """Check wrap_predict_method output with custom name."""

    target = """
def renamed_wrapper(a, b):
    "Output: c, msg"
    result = None
    msg = None
    try:
        global _compile_error
        if _compile_error is not None:
            raise _compile_error
        import numpy as np
        import pandas as pd

        if a is None: a = np.nan
        if b is None: b = np.nan
        input_array = np.array([a, b]).reshape((1, -1))
        columns = ["a", "b"]
        input_df = pd.DataFrame(data=input_array, columns=columns)
        result = dummy_func(input_df)
        result = tuple(result.ravel()) if hasattr(result, "ravel") else tuple(result)
        if len(result) == 0:
            result = tuple(None for i in range(1))
        elif "numpy" in str(type(result[0])):
            result = tuple(np.asscalar(i) for i in result)
    except Exception as e:
        from traceback import format_exc
        msg = str(e) + format_exc()
        if result is None:
            result = tuple(None for i in range(1))
    return result + (msg, )
        """.rstrip()

    code = build_wrapper_function(
        dummy_func,
        [
            DS2Variable('a', float, False),
            DS2Variable('b', float, False),
            DS2Variable('c', float, True),
        ],
        array_input=True,
        name='renamed_wrapper',
    )

    assert code == target


def test_wrap_predict_method():
    """Check wrap_predict_method output with default inputs."""
    from sasctl.utils.pymas.core import wrap_predict_method

    target = """
def predict(a, b):
    "Output: c, msg"
    result = None
    msg = None
    try:
        global _compile_error
        if _compile_error is not None:
            raise _compile_error
        import numpy as np
        import pandas as pd

        if a is None: a = np.nan
        if b is None: b = np.nan
        input_array = np.array([a, b]).reshape((1, -1))
        columns = ["a", "b"]
        input_df = pd.DataFrame(data=input_array, columns=columns)
        result = dummy_func(input_df)
        result = tuple(result.ravel()) if hasattr(result, "ravel") else tuple(result)
        if len(result) == 0:
            result = tuple(None for i in range(1))
        elif "numpy" in str(type(result[0])):
            result = tuple(np.asscalar(i) for i in result)
    except Exception as e:
        from traceback import format_exc
        msg = str(e) + format_exc()
        if result is None:
            result = tuple(None for i in range(1))
    return result + (msg, )
        """.rstrip()

    code = wrap_predict_method(
        dummy_func,
        [
            DS2Variable('a', float, False),
            DS2Variable('b', float, False),
            DS2Variable('c', float, True),
        ],
    )

    assert code == target


def test_wrap_predict_proba_method():
    """Check wrap_predict_proba_method output with default inputs."""
    from sasctl.utils.pymas.core import wrap_predict_proba_method

    target = """
def predict_proba(a, b):
    "Output: c, msg"
    result = None
    msg = None
    try:
        global _compile_error
        if _compile_error is not None:
            raise _compile_error
        import numpy as np
        import pandas as pd

        if a is None: a = np.nan
        if b is None: b = np.nan
        input_array = np.array([a, b]).reshape((1, -1))
        columns = ["a", "b"]
        input_df = pd.DataFrame(data=input_array, columns=columns)
        result = dummy_func(input_df)
        result = tuple(result.ravel()) if hasattr(result, "ravel") else tuple(result)
        if len(result) == 0:
            result = tuple(None for i in range(1))
        elif "numpy" in str(type(result[0])):
            result = tuple(np.asscalar(i) for i in result)
    except Exception as e:
        from traceback import format_exc
        msg = str(e) + format_exc()
        if result is None:
            result = tuple(None for i in range(1))
    return result + (msg, )
        """.rstrip()

    code = wrap_predict_proba_method(
        dummy_func,
        [
            DS2Variable('a', float, False),
            DS2Variable('b', float, False),
            DS2Variable('c', float, True),
        ],
    )

    assert code == target
