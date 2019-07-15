from .. import core
from ..core import sasctl_command, HTTPError


class Service(object):
    _SERVICE_ROOT = None

    is_uuid = core.is_uuid
    get_link = core.get_link
    request_link = core.request_link

    @classmethod
    def is_available(cls):
        """Checks if the service is currently available.

        Returns
        -------
        bool

        """
        response = cls.head('/')
        return response.status_code == 200

    @classmethod
    def request(cls, verb, path, session=None, raw=False, **kwargs):
        session = session or core.current_session()

        if session is None:
            raise TypeError('No `Session` instance found.')

        if path.startswith('/'):
            path = cls._SERVICE_ROOT + path
        else:
            path = cls._SERVICE_ROOT + '/' + path

        response = session.request(verb, path, **kwargs)

        if 400 <= response.status_code <= 599:
            raise HTTPError(response.url, response.status_code,
                            response.text, response.headers, None)
        try:
            if raw:
                return response.json()
            else:
                obj = core._unwrap(response.json())

                # ETag is required to update any object
                # May not be returned on all responses (e.g. listing
                # multiple objects)
                if isinstance(obj, core.RestObj):
                    setattr(obj, '_headers', response.headers)
                return obj
        except ValueError:
            return response.text

    @classmethod
    def get(self, *args, **kwargs):
        try:
            return self.request('get', *args, **kwargs)
        except HTTPError as e:
            if e.code == 404:
                return None  # Resource not found
            else:
                raise e

    @classmethod
    def head(cls, *args, **kwargs):
        return cls.request('head', *args, **kwargs)

    @classmethod
    def post(cls, *args, **kwargs):
        return cls.request('post', *args, **kwargs)

    @classmethod
    def _list_function(cls, path, single_term, plural_term, service_name):
        @sasctl_command('list')
        def list_items(filter=None):
            """List all {items} available in the environment.

            Parameters
            ----------
            filter : str, optional

            Returns
            -------
            list
                A list of dictionaries containing the {items}.

            Notes
            -----
            See the filtering_ reference for details on the `filter` parameter.

            .. _filtering: https://developer.sas.com/reference/filtering/

            """
            params = 'filter={}'.format(filter) if filter is not None else {}

            results = cls.get(path, params=params)
            return results if isinstance(results, list) else [results]

        list_items.__doc__ = list_items.__doc__.format(item=single_term,
                                                       items=plural_term)
        list_items._cli_service = service_name
        list_items.__name__ = 'list_%s' % plural_term
        return list_items

    @classmethod
    def _get_function(cls, path, single_term, plural_term, service_name,
                      list_func):
        @sasctl_command('get')
        def get_item(item, refresh=False):
            """Returns a {item} instance.

            Parameters
            ----------
            item : str or dict
                Name, ID, or dictionary representation of the {item}.
            refresh : bool, optional
                Obtain an updated copy of the {item}.

            Returns
            -------
            RestObj or None
                A dictionary containing the {item} attributes or None.

            Notes
            -------
            If `item` is a complete representation of the {item} it will be
            returned unless `refresh` is set.  This prevents unnecessary REST
            calls when data is already available on the client.

            """

            # If the input already appears to be the requested object just
            # return it, unless a refresh of the data was explicitly requested.
            if isinstance(item, dict) and all(
                    [k in item for k in ('id', 'name')]):
                if refresh:
                    item = item['id']
                else:
                    return item

            if cls.is_uuid(item):
                return cls.get(path + '/{id}'.format(id=item))
            else:
                results = list_func(filter='eq(name, "{}")'.format(item))

                # Not sure why, but as of 19w04 the filter doesn't seem to work.
                for result in results:
                    if result['name'] == str(item):
                        # Make a request for the specific object so that ETag
                        # is included, allowing updates.
                        if cls.get_link(result, 'self'):
                            return cls.request_link(result, 'self')
                        else:
                            id = result.get('id', result['name'])
                        return cls.get(path + '/{id}'.format(id=id))

                return None

            assert item is None or isinstance(item, dict)
            return item
        get_item.__doc__ = get_item.__doc__.format(item=single_term,
                                                   items=plural_term)
        get_item._cli_service = service_name
        get_item.__name__ = 'get_%s' % single_term
        return get_item

    @classmethod
    def _update_function(cls, path, single_term, plural_term, service_name):
        @sasctl_command('update')
        def update_item(item):
            """Updates a {item} instance.

            Parameters
            ----------
            item : dict

            Returns
            -------
            None

            """

            headers = getattr(item, '_headers', None)
            if headers is None or headers.get('etag') is None:
                raise ValueError(
                    'Could not find ETag for update of %s.' % item)

            id = getattr(item, 'id', None)
            if id is None:
                raise ValueError(
                    'Could not find property `id` for update of %s.' % item)

            headers = {'If-Match': item._headers.get('etag'),
                       'Content-Type': item._headers.get('content-type')}

            return cls.put(path + '/%s' % id, json=item, headers=headers)
        update_item.__doc__ = update_item.__doc__.format(item=single_term,
                                                   items=plural_term)
        update_item._cli_service = service_name
        update_item.__name__ = 'get_%s' % single_term
        return update_item

    @classmethod
    def _delete_function(cls, path, single_term, plural_term, service_name,
                         get_func):
        @sasctl_command('delete')
        def delete_item(item):
            """Deletes a {item} instance.

            Parameters
            ----------
            item

            Returns
            -------
            None

            """

            # Try to find the item if the id can't be found
            if not (isinstance(item, dict) and 'id' in item):
                item = get_func(item)

            if isinstance(item, dict) and 'id' in item:
                item = item['id']

            if cls.is_uuid(item):
                return cls.delete(path + '/{id}'.format(id=item))
            else:
                raise ValueError("Unrecognized id '%s'" % item)
        delete_item.__doc__ = delete_item.__doc__.format(item=single_term,
                                                   items=plural_term)
        delete_item._cli_service = service_name
        delete_item.__name__ = 'get_%s' % single_term
        return delete_item

    @classmethod
    def _crud_funcs(cls, path, single_term=None, plural_term=None,
                    service_name=None, list_func=True, get_func=True,
                    update_func=True, delete_func=True):
        """Utility method for defining simple functions to perform CRUD
        operations on a REST endpoint.

        Parameters
        ----------
        path : str
            URL path to use for the requests
        single_term : str
            English name of the item being manipulated.
            Defaults to `plural_term`.
        plural_term : str
            English name of the items being manipulated. Defaults to the last
            segment of `path`.
        service_name : str
            Name of the service under which the command will be listed in
            the `sasctl` CLI.  Defaults to `plural_term`.

        Returns
        -------
        functions : tuple
            tuple of CRUD functions: list_items, get_item, update_item, delete_item

        Examples
        --------

        >>> list_spam, get_spam, update_spam, delete_spam = _build_crud_funcs('/spam')

        """
        # Pull object name from path if unspecified (many paths end in /folders
        # or /repositories).
        plural_term = plural_term or str(path).split('/')[-1]
        single_term = single_term or plural_term
        service_name = service_name or plural_term
        service_name = service_name.replace(' ', '_')
        functions = []

        if list_func:
            functions.append(cls._list_function(path, single_term,
                                                plural_term, service_name))
        if get_func:
            try:
                _list = functions[0]
            except IndexError:
                _list = lambda x: []

            functions.append(cls._get_function(path, single_term,
                                               plural_term, service_name,
                                               _list))
        if update_func:
            functions.append(cls._update_function(path, single_term,
                                                  plural_term, service_name))
        if delete_func:
            if list_func and get_func:
                _get = functions[1]
            else:
                _get = lambda x: []
            functions.append(cls._delete_function(path, single_term,
                                                  plural_term, service_name,
                                                  _get))
        return tuple(functions)
