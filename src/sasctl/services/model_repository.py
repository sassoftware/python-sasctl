#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from sasctl.core import is_uuid, get, post, get_link, _build_crud_funcs, current_session


ROOT_PATH = '/modelRepository'
FUNCTIONS = {'Analytical', 'Classification', 'Clustering', 'Forecasting', 'Prediction', 'Text categorization',
             'Text extraction', 'Text sentiment', 'Text topics', 'Sentiment'}

# TODO:  automatic query string encoding

list_repositories, get_repository, update_repository, delete_repository = _build_crud_funcs(ROOT_PATH + '/repositories',
                                                                                            'repository')
list_projects, get_project, update_project, delete_project = _build_crud_funcs(ROOT_PATH + '/projects', 'project')
list_models, get_model, update_model, delete_model = _build_crud_funcs(ROOT_PATH + '/models', 'model')


def get_model_link(model, rel, refresh=False):
    """Retrieve a link from a model's set of links.

    Parameters
    ----------
    model : str or dict
        The name or id of the model, or a dictionary representation of the model.
    rel, str
        The name of the link to retrieve
    refresh, bool
        Whether the model's data should be refreshed before searching for the link.

    Returns
    -------
    dict or None
        Dictionary containing the link's properties

    """

    # Try to find the link if the model is a dictionary
    if isinstance(model, dict):
        link = get_link(model, rel)
        if link is not None or not refresh:
            return link

    # Model either wasn't a dictionary or it didn't include the link
    # Pull the model data and then check for the link
    model = get_model(model, refresh=refresh)

    return get_link(model, rel)


def get_astore(model):
    # TODO: Download binary object?

    link = get_model_link(model, 'analyticStore', refresh=True)

    if link is not None:
        return link.get('href')


def get_score_code(model):
    """The score code for a model registered in the model repository.

    Parameters
    ----------
    model : str or dict
        The name or id of the model, or a dictionary representation of the model.

    Returns
    -------
    str
        The code designated as the model's score code

    """

    link = get_model_link(model, 'scoreCode', refresh=True)

    if link is not None:
        scorecode_uri = link.get('href')
        return get(scorecode_uri, headers={'Accept': 'text/vnd.sas.models.score.code.ds2package'})


def get_model_contents(model):
    """The additional files and data associated with the model.

    Parameters
    ----------
    model : str or dict
        The name or id of the model, or a dictionary representation of the model.

    Returns
    -------
    list
        The code designated as the model's score code

    """

    link = get_model_link(model, 'contents', refresh=True)

    if link is not None:
        return get(link.get('href'), headers={'Accept': link.get('itemType', '*/*')})


def create_model(model, project, description=None, modeler=None, function=None, algorithm=None, tool=None,
                 is_champion=False,
                 properties={}, **kwargs):
    """

    Parameters
    ----------
    model
    project
    description : str, optional
    modeler : str, optional
        Name of the user that created the model.  Current user name will be used if unspecified.

    function
    algorithm
    tool
    modeler
    scoreCodeType
    trainTable
    classificationEventProbabilityVariableName
    classificationTargetEventValue
    champion (T/F)
    role
    location
    targetVariable
    projectId, projectName, projectVersionId, projectVersionName???
    suggestedChampion (T/F)
    retrainable
    immutable
    modelVersionName
    properties  (custom properties)
        name
        value
        type
    inputVariables
        -
    outputVariables
        -

    properties
    kwargs

    Returns
    -------

    """

    if isinstance(model, str):
        model = {'name': model}

    assert isinstance(model, dict)

    p = get_project(project)
    if p is None:
        raise ValueError("Unable to find project '%s'" % project)

    model['projectId'] = p['id']
    model['modeler'] = modeler or current_session().user

    model['description'] = description or model.get('description')
    model['function'] = function or model.get('function')
    model['algorithm'] = algorithm or model.get('algorithm')
    model['tool'] = tool or model.get('tool')
    model['champion'] = is_champion or model.get('champion')
    model['role'] = 'Champion' if model.get('champion', False) else 'Challenger'
    model['description'] = description or model.get('description')
    model.setdefault('properties', [{'name': k, 'value': v} for k, v in properties.items()])

# TODO: add kwargs (pop)
#     model.update(kwargs)
    return post(ROOT_PATH + '/models', json=model, headers={'Content-Type': 'application/vnd.sas.models.model+json'})


def add_model_content(model, file, name=None, role=None):
    if is_uuid(model):
        id = model
    elif isinstance(model, dict) and 'id' in model:
        id = model['id']
    else:
        model = get_model(model)
        id = model['id']

    metadata = {'role': role}
    if name is not None:
        metadata['name'] = name

    return post(ROOT_PATH + '/models/{}/contents'.format(id), files={name: file}, data=metadata)


def default_repository():
    """Get the built-in default repository.

    Returns
    -------
    RestObj

    """

    repo = get_repository('Repository 1')   # Default in 19w04
    if repo is None:
        repo = get_repository('Public')     # Default in 19w21
    if repo is None:
        repo = list_repositories()[0]

    return repo


def create_project(project, repository, **kwargs):
    if isinstance(project, str):
        project = {'name': project}

    repository = get_repository(repository)

    project['repositoryId'] = repository['id']
    project['folderId'] = repository['folderId']

    project.update(kwargs)
    return post(ROOT_PATH + '/projects', json=project, headers={'Content-Type': 'application/vnd.sas.models.project+json'})


def import_model_from_zip(name, project, file, description=None, version='latest'):
    # TODO: Allow import into folder if no project is given
    # TODO: Create new version if model already exists
    project = get_project(project)

    if project is None:
        raise ValueError('Project `%s` could not be found.' % str(project))

    params = {'name': name,
              'description': description,
              'type': 'ZIP',
              'projectId': project.id,
              'versionOption': version}
    params = '&'.join(['{}={}'.format(k, v) for k, v in params.items()])

    r = post(ROOT_PATH + '/models#octetStream',
                data=file.read(),
                params=params,
                headers={'Content-Type': 'application/octet-stream'})

    return r


