#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from .service import Service
from sasctl.core import HTTPError


class MicroAnalyticScore(Service):
    """A stateless, memory-resident, high-performance program execution
    service.

    """
    _SERVICE_ROOT = '/microanalyticScore'

    def list_modules(self, filter=None):
        params = 'filter={}'.format(filter) if filter is not None else {}

        return self.get('/modules', params=params)

    def get_module(self, module):
        if isinstance(module, dict) and all([k in module for k in ('id', 'name')]):
            return module

        try:
            # MAS module IDs appear to just be the lower case name.
            # Try to find by ID first
            return self.get('/modules/{}'.format(str(
                module).lower()))
        except HTTPError as e:
            if e.code == 404:
                pass
            else:
                raise e

        # Wasn't able to find by id, try searching by name
        results = self.list_modules(filter='eq(name, "{}")'.format(module))

        # Not sure why, but as of 19w04 the filter doesn't seem to work.
        for result in results:
            if result['name'] == str(module):
                return result

    def get_module_step(self, module, step):
        module = self.get_module(module)

        r = self.get('/modules/{}/steps/{}'.format(module.id, step))
        return r

    def list_module_steps(self, module):
        module = self.get_module(module)

        return self.get('/modules/{}/steps'.format(module.id))

    def execute_module_step(self, module, step, return_dict=True, **kwargs):
        module_name = module.name if hasattr(module, 'name') else str(module)
        module = self.get_module(module)

        if module is None:
            raise ValueError("Module '{}' was not found.".format(module_name))
        module = module.id
        step = step.id if hasattr(step, 'id') else step

        body = {'inputs': [{'name': k, 'value': v} for k, v in kwargs.items()]}
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
        """

        Parameters
        ----------
        name : str
        description : str
        source : str
        language : str { 'python', 'ds2' }
        scope : str { 'public', 'private' }

        Returns
        -------

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
            raise ValueError('Unrecognized source code language `%s`.' % language)

        data = {'id': name,
                'type': t,
                'description': description,
                'source': source,
                'scope': scope}

        r = self.post('/modules', json=data)
        return r

    def define_steps(self, module):
        import types

        module = self.get_module(module)

        for id in module.get('stepIds', []):
            step = self.get_module_step(module, id)

            name = '_{}_{}'.format(module.name, step.id)
            arguments = [k['name'] for k in step.inputs]
            arg_types = [k['type'] for k in step.inputs]
            call_params = ['{}={}'.format(i, i) for i in arguments]
            type_string = '    # type: ({})'.format(', '.join(arg_types))

            code = ('def {}({}):'.format(name, ', '.join(arguments)),
                    type_string,
                    '    """docstring"""',
                    '    return execute_module_step(module, step, return_dict=False, {})'.format(', '.join(call_params))
                    )

            code = '\n'.join(code)
            compiled = compile(code, '<string>', 'exec')

            env = globals().copy()
            env.update({'execute_module_step': self.execute_module_step,
                        'module': module,
                        'step': step})

            func = types.FunctionType(compiled.co_consts[0], env)

            setattr(module, step.id, func)

        return module
