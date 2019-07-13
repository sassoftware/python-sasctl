#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import uuid
from collections import namedtuple, OrderedDict

import six


class DS2Package(object):
    def __init__(self, variables, code=None, return_code=True,
                 return_message=True, target=None):
        self._id = uuid.uuid4().hex.upper()
        self._python_code = code or []
        code = code or []

        self.methods = [DS2PyMASMethod(variables, code, return_code,
                                       return_message, target)]

        self._body = ("dcl package pymas py;",
                      "dcl package logger logr('App.tk.MAS');",
                      "dcl varchar(67108864) character set utf8 pycode;",
                      "dcl int revision;")

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        # Max length in some SAS products in 32 characters
        return ('_' + str(self.id))[:32]

    def code(self):
        code = ("package %s / overwrite=yes;" % self.name,) + \
               tuple('    ' + line for line in self._body) + \
               ('',)

        for method in self.methods:
            code += tuple('    ' + line for line in method.code().split('\n'))
        code += ("endpackage;",
                 '')

        return '\n'.join(code)


class DS2BaseMethod(object):
    def __init__(self, name, variables, body=None):
        self._name = name
        self.variables = variables

        if body is None:
            self._body = []
        elif isinstance(body, six.string_types):
            self._body = body.split('\n')
        else:
            self._body = list(body)

    @property
    def name(self):
        return self._name

    def code(self):
        vars = ',\n'.join(
            ['    %s' % v.as_parameter() for v in self.variables])

        # Don't spread signature over multiple lines if there are no variables
        if len(vars) > 0:
            func = ('method %s(' % self.name,
                    vars,
                    '    );',
                    '')
        else:
            func = ('method %s();' % self.name,
                    '')

        if len(self._body) > 0:
            func += tuple('    ' + line for line in self._body)

        func += ('end;',
                 '')

        return '\n'.join(func)


class DS2PyMASMethod(DS2BaseMethod):
    def __init__(self, variables, python_code, return_code=True,
                 return_message=True, target='wrapper'):

        target = target or 'wrapper'
        if isinstance(python_code, six.string_types):
            python_code = python_code.split('\n')

        self.public_variables = variables
        self.private_variables = []

        if return_code:
            self.public_variables.append(DS2Variable('rc', 'int', True))
        else:
            self.private_variables.append(DS2Variable('rc', 'int', True))

        if return_message:
            self.public_variables.append(DS2Variable('msg', 'char', True))

        body = [v.as_declaration() for v in
                self.private_variables] \
               + ["if null(py) then do;",
                  "    py = _new_ pymas();",
                  "    rc = py.useModule('mypymodule', 1);",
                  "    if rc then do;"] \
               + ["        rc = py.appendSrcLine('%s');" % l for l in
                  python_code] \
               + ["        pycode = py.getSource();",
                  "        revision = py.publish(pycode, 'mypymodule');",
                  "        if revision lt 1 then do;",
                  "            logr.log('e', 'py.publish() failed.');",
                  "            rc = -1;",
                  "            return;",
                  "        end;",
                  "    end;",
                  "    rc = py.useMethod('%s');" % target,
                  "    if rc then return;",
                  "end;"] \
               + ['%s    if rc then return;' % v.pymas_statement() for v in
                  self.public_variables if not v.out] \
               + ['rc = py.execute();    if rc then return;'] \
               + [v.pymas_statement() for v in self.public_variables
                  if v.out and v.name != 'rc']

        super(DS2PyMASMethod, self).__init__('score', variables,
                                             body=body)


class DS2ScoreMethod(DS2BaseMethod):
    def __init__(self, variables, return_code=True, return_message=True,
                 target='wrapper', ):
        self._target = target

        self.public_variables = variables
        self.private_variables = []

        if return_code:
            self.public_variables.append(DS2Variable('rc', 'int', True))
        else:
            self.private_variables.append(DS2Variable('rc', 'int', True))

        if return_message:
            self.public_variables.append(DS2Variable('msg', 'char', True))
        else:
            self.private_variables.append(DS2Variable('msg', 'char', True))

        body_statements = [v.as_declaration() for v in
                           self.private_variables] + \
                          [v.pymas_statement() + '    if rc then return;' for v
                           in self.public_variables if
                           not v.out] + \
                          ['rc = py.execute();    if rc then return;'] + \
                          [v.pymas_statement() for v in self.public_variables
                           if v.out and v.name != 'rc']

        super(DS2ScoreMethod, self).__init__('score', variables,
                                             body=body_statements)


class DS2Method(object):
    def __init__(self, variables, code, target='wrapper'):
        self.variables = variables
        self._code = code
        self.target = target

    @property
    def name(self):
        return 'score'

    def code(self, return_code=True, return_message=True):
        signature = [v.as_parameter() for v in self.variables]

        if return_code:
            signature += ['in_out double rc']
        if return_message:
            signature += ['in_out char msg']

        signature = ', '.join(signature)

        # Python code to be embedded passed to PyMAS
        code = [self._code] if isinstance(self._code, str) else self._code
        code = ["\t\t\t\t\terr = py.appendSrcLine('{}');".format(line) for line
                in code]
        code = '\n'.join(code)

        set_statements = [v.pymas_statement() for v in self.variables if
                          not v.out]
        get_statements = [v.pymas_statement() for v in self.variables if v.out]

        if return_code:
            get_statements += ['rc = err;']
        if return_message:
            get_statements += ["msg = py.getString('msg');"]

        ds2_statements = set_statements + [
            'err = py.execute();'] + get_statements
        ds2_statements = '\n'.join(['\t\t\t\t' + s for s in ds2_statements])

        func = ('            method {}({});'.format(self.name, signature),
                '                dcl double err;',
                '                if null(py) then do;',
                '                    py = _new_ pymas();',
                code,
                '                    pycode = py.getSource();',
                "                    revision = py.publish( pycode, 'mypymodule' );",
                "                    err = py.useMethod('{}');".format(
                    self.target),
                "                    if err then return;",
                "                end;",
                ds2_statements,
                "            end;")
        func = '\n'.join(func)

        return func


class DS2Thread(object):
    def __init__(self, variables, table, column_names=None, return_code=True,
                 return_message=True, package=None):
        """
        Args:
            variables:
            table:
            return_code:
            return_message:
            name:
        """

        self._id = uuid.uuid4().hex.upper()
        self.table = table
        self.variables = variables
        self.return_code = return_code
        self.return_message = return_message
        self.column_names = column_names
        self.package = package

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return 'pyMasThread'

    def __str__(self):
        extras = ['"rc"'] if self.return_code else []
        extras += ['"msg"'] if self.return_message else []

        array_input = any([v.is_array for v in self.variables if not v.out])

        # If passing column data into Python as an array, need extra assignment statements to set values
        if array_input and self.column_names is not None:
            array = next(filter(lambda v: v.is_array, self.variables))
            var_assignments = [
                '{}[{}] = {};'.format(array.name, i + 1, self.column_names[i])
                for i in
                range(min(array.size, len(self.column_names)))]
            var_assignments = '\n'.join(var_assignments)
        else:
            var_assignments = ''

        # Declare output variables.  Input variables are assumed to be columns in the input table.
        declarations = '\n'.join([v.as_declaration() for v in self.variables if
                                  v.out or v.is_array])

        keep_vars = [v.name for v in self.variables] + extras

        code = ("        thread {} / inline;".format(self.name),
                "            dcl package {} decisionPackage();".format(
                    self.package.name),
                declarations,
                '            dcl double "rc";',
                "            dcl char(4096) character set utf8 msg;",
                # "            keep {};".format(' '.join(keep_vars)),
                "            method run();",
                "                dcl integer localRC;",
                "                set {} ( );".format(self.table),
                var_assignments,
                "                decisionPackage.{}({});".format(
                    self.package.methods[0].name,
                    ','.join(keep_vars)),
                "                output;",
                "            end;",
                "        endthread;")

        code = '\n'.join(code)

        return code


class DS2Variable(namedtuple('Ds2Variable', ['name', 'type', 'out'])):
    PY_TYPE_TO_DS2 = OrderedDict([('double64', 'double'),
                                  ('double32', 'double'),
                                  ('double', 'double'),
                                  # Terminates search if full string matches
                                  ('float64', 'double'),
                                  ('float32', 'double'),
                                  ('float', 'double'),
                                  ('string', 'char'),
                                  ('str', 'char'),
                                  ('varchar', 'char'),
                                  ('integer64', 'integer'),
                                  ('integer32', 'integer'),
                                  ('integer', 'integer'),
                                  # Terminates search if full string matches
                                  ('int64', 'integer'),
                                  ('int32', 'integer'),
                                  ('int', 'integer')
                                  ])

    DS2_TYPE_TO_VIYA = OrderedDict([('double', 'decimal'),
                                    ('varchar', 'string'),
                                    ('char', 'string')
                                    ])

    def __new__(cls, *args, **kwargs):

        # Convert Python types to DS2 types if necessary
        if 'type' in kwargs:
            kwargs['type'] = DS2Variable._map_type(cls.PY_TYPE_TO_DS2,
                                                   kwargs['type'])
        elif len(args) > 1:
            args = list(args)
            args[1] = DS2Variable._map_type(cls.PY_TYPE_TO_DS2, args[1])

        return super(DS2Variable, cls).__new__(cls, *args, **kwargs)

    @classmethod
    def _map_type(cls, mapping, t):
        # Convert Python type names to DS2 type names
        t = str(t).lower().strip()

        # Using replace since type could be an array: float[10]
        for k, v in six.iteritems(mapping):
            if t.startswith(k):
                t = t.replace(k, v)
                break

        return t

    def as_model_metadata(self):
        viya_type = self._map_type(self.DS2_TYPE_TO_VIYA, self.type)
        role = 'Output' if self.out else 'Input'

        return OrderedDict(
            [('name', self.name), ('role', role), ('type', viya_type)])

    def as_declaration(self):
        """DS2 variable declaration statement."""
        match = re.search('\[\d+\]$', self.type)
        if match is None:
            return 'dcl {} {};'.format(self.type, self.name)
        else:
            # Type is an array
            return 'dcl {} {}{};'.format(self.type[:match.start()], self.name,
                                         self.type[match.start():])

    def as_parameter(self):
        """DS2 parameter syntax for method signatures."""
        match = re.search('\[\d+\]$', self.type)
        param = self.name

        param = self.type + ' ' + param if match is None else self.type[
                                                              :match.start():] + ' ' + param + '[*]'
        param = 'in_out ' + param if self.out else param

        return param

    @property
    def size(self):
        match = re.search('\[\d+\]$', self.type)
        if match:
            return int(self.type[match.start() + 1:match.end() - 1])
        else:
            return 0

    def pymas_statement(self, python_var_name=None, pymas_name='py'):
        """Returns appropriate PyMAS get/set statements to move values between a Python variable and the DS2 variable.

        Args:
            python_var_name: Python variable name.
            pymas_name: Name of the DS2 PyMAS package being used.  Defaults to `py`

        Returns:

        """

        # Default Python variable name to DS2 variable name
        python_var_name = python_var_name or self.name

        if self.out:
            if self.type.startswith('double'):
                if self.is_array:
                    return "py.getDoubleArray('{}', {}, ret);".format(
                        python_var_name, self.name)
                else:
                    return "{} = py.getDouble('{}');".format(self.name,
                                                             python_var_name)
            elif self.type.startswith('char'):
                if self.is_array:
                    return "py.getStringArray('{}', {}, ret);".format(
                        python_var_name, self.name)
                else:
                    return "{} = py.getString('{}');".format(self.name,
                                                             python_var_name)
            elif self.type.startswith('integer'):
                if self.is_array:
                    return "py.getIntArray('{}', {}, ret);".format(
                        python_var_name, self.name)
                else:
                    return "{} = py.getInt('{}');".format(self.name,
                                                          python_var_name)
            else:
                raise ValueError(
                    "Can't generate a DS2 statement for type `{}`".format(
                        self.type))
        else:
            if self.type.startswith('double'):
                if self.is_array:
                    return "rc = py.setDoubleArray('{}', {});".format(
                        python_var_name, self.name)
                else:
                    return "rc = py.setDouble('{}', {});".format(
                        python_var_name, self.name)
            elif self.type.startswith('char'):
                if self.is_array:
                    return "rc = py.setStringArray('{}', {});".format(
                        python_var_name, self.name)
                else:
                    return "rc = py.setString('{}', {});".format(
                        python_var_name, self.name)
            elif self.type.startswith('integer'):
                if self.is_array:
                    return "rc = py.setIntArray('{}', {});".format(
                        python_var_name, self.name)
                else:
                    return "rc = py.setInt('{}', {});".format(python_var_name,
                                                              self.name)

            else:
                raise ValueError(
                    "Can't generate a DS2 statement for type `{}`".format(
                        self.type))

    @property
    def is_array(self):
        # If type contains [n] then it's an array
        return re.search('\[\d+\]$', self.type) is not None
