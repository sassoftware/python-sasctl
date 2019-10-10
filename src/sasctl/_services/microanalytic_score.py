#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stateless, memory-resident, high-performance program execution service."""

import re
from collections import OrderedDict
from math import isnan

import six

from .service import Service


class MicroAnalyticScore(Service):
    """Micro Analytic Service (MAS) client."""

    _SERVICE_ROOT = '/microanalyticScore'

    @classmethod
    def is_uuid(cls, id):
        """Check if the ID appears to be a valid MAS id.

        Indicates whether `id` appears to be a correctly formatted ID.  Does
        **not** check whether a module with `id` actually exists.

        Parameters
        ----------
        id : str

        Returns
        -------
        bool

        Notes
        -----
        Overrides the :meth:`Service.is_uuid` method since MAS modules do
        not currently use IDs that are actually UUIDs.

        """
        # Anything that consists of only numbers, lowercase letters,
        # and underscores, and does not start with a number, looks like a
        # MAS id.
        return re.match('^[_a-z][_a-z0-9]+$', id) is not None

    list_modules, get_module, update_module, \
        delete_module = Service._crud_funcs('/modules', 'module')

    def get_module_step(self, module, step):
        """Details of a single step in a given module.

        Parameters
        ----------
        module : str or dict
            Name, id, or dictionary representation of a module
        step : str
            Name of the step

        Returns
        -------
        RestObj

        """
        module = self.get_module(module)

        r = self.get('/modules/{}/steps/{}'.format(module.id, step))
        return r

    def list_module_steps(self, module):
        """List all steps defined for a module.

        Parameters
        ----------
        module : str or dict
            Name, id, or dictionary representation of a module

        Returns
        -------
        list
            List of :class:`.RestObj` instances representing each step.

        """
        module = self.get_module(module)

        steps = self.get('/modules/{}/steps'.format(module.id))
        return steps if isinstance(steps, list) else [steps]

    def execute_module_step(self, module, step, return_dict=True, **kwargs):
        """Call a module step with the given parameters.

        Parameters
        ----------
        module : str or dict
            Name, id, or dictionary representation of a module
        step : str
            Name of the step
        return_dict : bool, optional
            Whether the results should be returned as a dictionary instead
            of a tuple
        kwargs : any
            Passed as arguments to the module step

        Returns
        -------
        any
            Results of the step execution.  Returned as a dictionary if
            `return_dict` is True, otherwise returned as a tuple if more
            than one value is returned, otherwise the single value.

        """
        module_name = module.name if hasattr(module, 'name') else str(module)
        module = self.get_module(module)

        if module is None:
            raise ValueError("Module '{}' was not found.".format(module_name))
        module = module.id
        step = step.id if hasattr(step, 'id') else step

        # Make sure all inputs are JSON serializable
        # Common types such as numpy.int64 and numpy.float64 are NOT serializable
        for k in kwargs.keys():
            type_name = type(kwargs[k]).__name__
            if type_name == 'float64':
                kwargs[k] = float(kwargs[k])
            elif type_name == 'int64':
                kwargs[k] = int(kwargs[k])

        body = {'inputs': [{'name': k, 'value': v}
                           for k, v in six.iteritems(kwargs)]}

        # Convert NaN to None (null) before calling MAS
        for input in body['inputs']:
            try:
                if isnan(input['value']):
                    input['value']  = None
            except TypeError:
                pass

        r = self.post('/modules/{}/steps/{}'.format(module, step), json=body)

        # Convert list of name/value pair dictionaries to single dict
        outputs = OrderedDict()
        for output in r.get('outputs', []):
            k, v = output['name'], output.get('value')

            # Remove padding from CHAR columns
            if isinstance(v, str):
                v = v.strip()

            outputs[k] = v

        if return_dict:
            # Return results as k=v pairs
            return outputs
        else:
            # Return only the values, as if calling another Python function.
            outputs = tuple(outputs.values())
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs

    def create_module(self, name=None, description=None, source=None,
                      language='python', scope='public'):
        """Create a new module in MAS.

        Parameters
        ----------
        name : str
        description : str
        source : str
        language : str { 'python', 'ds2' }
        scope : str { 'public', 'private' }

        Returns
        -------
        RestObj

        """
        if source is None:
            raise ValueError('The `source` parameter is required.')
        else:
            source = str(source)

        if language == 'python':
            t = 'text/x-python'
        elif language == 'ds2':
            t = 'text/vnd.sas.source.ds2'
        else:
            raise ValueError('Unrecognized source code language `%s`.'
                             % language)

        data = {'id': name,
                'type': t,
                'description': description,
                'source': source,
                'scope': scope}

        r = self.post('/modules', json=data)
        return r

    def define_steps(self, module):
        """Map MAS steps to Python methods.

        Defines python methods on a module that automatically call the
        corresponding MAS steps.

        Parameters
        ----------
        module : str or dict
            Name, id, or dictionary representation of a module

        Returns
        -------
        module

        """
        import types

        module = self.get_module(module)

        # Define a method for each step of the module
        for id in module.get('stepIds', []):
            step = self.get_module_step(module, id)

            # Method should have an argument for each parameter of the step
            arguments = [k['name'] for k in step.get('inputs', [])]
            arg_types = [k['type'] for k in step.get('inputs', [])]

            # Format call to execute_module_step()
            call_params = ['{}={}'.format(i, i) for i in arguments]

            # Set type hints for the function
            type_string = '    # type: ({})'.format(', '.join(arg_types))

            # Method signature
            input_params = [a for a in arguments] + ['**kwargs']
            signature = 'def _%s_%s(%s):' \
                        % (module.name,
                           step.id,
                           ', '.join(input_params))

            # MAS always lower-cases variable names
            # Since the original Python variables may have a different case,
            # allow kwargs to be used to input alternative caps
            if len(arguments):
                arg_checks = ['for k in kwargs.keys():']
                for arg in arguments:
                    arg_checks.append("    if k.lower() == '%s':" % arg.lower())
                    arg_checks.append("        %s = kwargs[k]" % arg)
                    arg_checks.append("        continue")
            else:
                arg_checks = []

            # Full method source code
            # Drops 'rc' and 'msg' from return values
            code = (signature,
                    type_string,
                    '    """Execute step %s of module %s."""' % (step, module),
                    '\n'.join(['    %s' % a for a in arg_checks]),
                    '    r = execute_module_step(%s)' % ', '.join(['module', 'step'] + call_params),
                    '    r.pop("rc", None)',
                    '    r.pop("msg", None)',
                    '    if len(r) == 1:',
                    '        return r.popitem()[1]',
                    '    return tuple(v for v in r.values())'
                    )

            code = '\n'.join(code)
            compiled = compile(code, '<string>', 'exec')

            env = globals().copy()
            env.update({'execute_module_step': self.execute_module_step,
                        'module': module,
                        'step': step})

            func = types.FunctionType(compiled.co_consts[0],
                                      env,
                                      argdefs=tuple(None for x in arguments))

            setattr(module, step.id, func)

        return module
