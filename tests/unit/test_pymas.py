#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from six.moves import mock
import pytest

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
    esp_code = p.score_code(dest='esp')

    with pytest.raises(ValueError):
        cas_code = p.score_code(dest='cas')

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
        assert [DS2Variable('x1', 'str', False),
                DS2Variable('x2', 'int', False),
                DS2Variable('out1', 'float', True)] == call_args[1]   # Variables

    with mock.patch('sasctl.utils.pymas.core.PyMAS', autospec=True) as mocked:
        _ = from_inline(dummy_func, input_types=int)
        assert 1 == mocked.call_count
        call_args = mocked.call_args[0]
        assert [DS2Variable('x1', 'int', False),
                DS2Variable('x2', 'int', False),
                DS2Variable('result', 'float', True)] == call_args[1]   # Variables

    with mock.patch('sasctl.utils.pymas.core.PyMAS', autospec=True) as mocked:
        _ = from_inline(dummy_func, input_types=OrderedDict([('a', int), ('b', float)]))
        assert 1 == mocked.call_count
        call_args = mocked.call_args[0]
        assert [DS2Variable('a', 'int', False),
                DS2Variable('b', 'double', False),
                DS2Variable('result', 'float', True)] == call_args[1]   # Variables


def test_from_pickle_with_func():
    """Create a PyMAS instance from a pickled object."""

    import pickle
    from sasctl.utils.pymas import from_pickle

    data = pickle.dumps(dummy_func)

    with mock.patch('sasctl.utils.pymas.core.PyMAS', autospec=True) as mocked:
        result = from_pickle(data)
        assert 1 == mocked.call_count
        call_args = mocked.call_args[0]
        assert [DS2Variable('x1', 'str', False),
                DS2Variable('x2', 'int', False),
                DS2Variable('out1', 'float', True)] == call_args[1]   # Variables

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
        assert [DS2Variable('x1', 'str', False),
                DS2Variable('x2', 'int', False),
                DS2Variable('out1', 'float', True)] == call_args[1]   # Variables

    assert isinstance(result, PyMAS)


def test_build_wrapper_function():
    def func(a, b):
        pass

    # Actual function inputs & DS2 variables current dont have to match.
    result = build_wrapper_function(func, [DS2Variable('a', 'int', False),
                                           DS2Variable('x', 'float', True)],
                                    array_input=False,
                                    return_msg=False)
    assert isinstance(result, str)
    assert '"Output: x"\n' in result


def test_ds2_double():
    # Input variable
    var = DS2Variable('myvar', 'float', False)
    assert 'double' == var.type
    assert False == var.is_array
    assert 'dcl double myvar;' == var.as_declaration()
    assert 'double myvar' == var.as_parameter()
    assert "err = py.setDouble('myvar', myvar);" == var.pymas_statement()

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
    assert "err = py.setString('myvar', myvar);" == var.pymas_statement()

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
    assert "err = py.setInt('myvar', myvar);" == var.pymas_statement()

    # Output variable
    var = DS2Variable('myvar', 'integer', True)
    assert 'integer' == var.type
    assert False == var.is_array
    assert 'dcl integer myvar;' == var.as_declaration()
    assert 'in_out integer myvar' == var.as_parameter()
    assert "myvar = py.getInt('myvar');" == var.pymas_statement()


def test_ds2_variables_dict_input():
    target = [DS2Variable(name='x1', type='double', out=False),
              DS2Variable(name='x2', type='double', out=False)]

    # Standard input
    result = ds2_variables(OrderedDict([('x1', ('double', False)), ('x2', ('double', False))]))
    assert result == target

    # No return value flag
    result = ds2_variables(OrderedDict([('x1', 'double'), ('x2', 'double')]))
    assert result == target


    assert [DS2Variable('a', 'int', False),
            DS2Variable('b', 'char', False),
            DS2Variable('c', 'double', False)] == ds2_variables(OrderedDict([('a', int), ('b', str), ('c', float)]))

    assert [DS2Variable('x', 'str', True)] == ds2_variables(OrderedDict(x=str), output_vars=True)

    assert [DS2Variable('a', 'int', False),
            DS2Variable('b', 'char', False),
            DS2Variable('x', 'double', True)] == ds2_variables(OrderedDict([('a', int),
                                                                            ('b', str),
                                                                            ('x', (float, True))]))


def test_ds2_variables_func_input():
    target = [DS2Variable(name='x1', type='double', out=False),
              DS2Variable(name='x2', type='double', out=False)]

    # Only function as input
    with mock.patch('sasctl.utils.pymas.python.parse_type_hints') as mocked:
        mocked.return_value = OrderedDict([('x1', ('double', False)), ('x2', ('double', False))])
        result = ds2_variables(lambda x: x)     # Input function doesn't matter since response is mocked
        assert 1 == mocked.call_count
    assert result == target


def test_ds2_variables_pandas_input():
    pd = pytest.importorskip('pandas')

    target = [DS2Variable('a', 'int', False),
              DS2Variable('b', 'char', False),
              DS2Variable('c', 'double', False)]

    df = pd.DataFrame(dict(a=[1,2,3], b=['1', '2', '3'], c=[1., 2., 3.]))
    df = df[['a', 'b', 'c']]    # Ensure columns are ordered correctly

    assert target == ds2_variables(df)


def test_ds2_variables_numpy_input():
    np = pytest.importorskip('numpy')


    assert [DS2Variable('var1', 'int', False),
            DS2Variable('var2', 'int', False),
            DS2Variable('var3', 'int', False)] == ds2_variables(np.array([1, 2, 3]))

    assert [DS2Variable('var1', 'double', False),
            DS2Variable('var2', 'double', False)] == ds2_variables(np.array([1., 2.]))



def test_parse_type_hints_source():
    def bad_func(x1, x2):
        return x1 + x2

    with pytest.raises(ValueError):
        parse_type_hints(bad_func)

    def func1(x1, x2):
        # type: (str, int) -> (float, bool)
        pass
    result = parse_type_hints(func1)
    assert result == OrderedDict([('x1', ('str', False)), ('x2', ('int', False)),
                                   ('out1', ('float', True)), ('out2', ('bool', True))])

    def func2(x1, x2):
        # type: float, float -> float
        pass
    result = parse_type_hints(func2)
    assert result == OrderedDict([('x1', ('float', False)), ('x2', ('float', False)), ('out1', ('float', True))])

    def func3(x1, x2):
        # type: float, float -> float, int
        pass
    result = parse_type_hints(func3)
    assert result == OrderedDict([('x1', ('float', False)), ('x2', ('float', False)),
                                  ('out1', ('float', True)), ('out2', ('int', True))])

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

