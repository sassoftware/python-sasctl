#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
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
    """Get the ASTORE for a model registered int he model repository.

    Parameters
    ----------
    model :  str or dict
        The name or id of the model, or a dictionary representation of the model.

    Returns
    ----------
    binary?

    """
    # TODO: Download binary object?

    link = get_model_link(model, 'analyticStore', refresh=True)

    if link is not None:
        return link.get('href')


def get_score_code(model):
    """Get the score code for a model registered in the model repository.

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
    """Retrieve the additional files and data associated with the model.

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


def create_model(model, project, description=None, modeler=None, function=None,
                 algorithm=None, tool=None, is_champion=False, properties={},
                 **kwargs):
    """Creates a model into a project or folder.

    Parameters
    ----------
    model : str or dict
        The name or id of the model, or a dictionary representation of the
        model.
    project : str or dict
        The name or id of the model project, or a dictionary representation of
        the model project.
    description : str, optional
        The description of the model.
    modeler : str, optional
        Name of the user that created the model.  Current user name will be
        used if unspecified.
    function : str, optional
        The function of the model, valid values include: analytical,
        classification, cluster, forecasting, prediction, Text analytics,
        transformation.
    algorithm : str, optional
        The name of the model algorithm.
    tool : str, optional
        The name of the model tool, can be 'Python 2' or 'Python 3'.
    scoreCodeType : str, optional
        The score code type for the model.
    trainTable : str, optional
        The train data table.
    classificationEventProbabilityVariableName : str, optional
        The name of the event probability variable.
    classificationTargetEventValue : str, optional
        The target event value.
    champion : bool, optional
        Indicates whether the project has champion model or not.
    role : str, optional
        The role of the model, valid values include: plain, champion,
        challenger.
    location : str, optional,
        The location of this model.
    targetVariable : str, optional
        The name of the target variable.
    suggestedChampion : bool
        Indicates the model was suggested as the champion at import time.
    retrainable : bool
        Indicates whether the model can be retrained or not.
    immutable : bool
        Indicates whether the model can be changed or not.
    modelVersionName : str, optional
        The display name for the model version.
    properties : array_like, optional (custom properties)
        Custom model properties that can be set: name, value, type

    inputVariables : array_like, optional
        Model input variables. By default, these are the same as the model
        project.
    outputVariables : array_like, optional
        Model output variables. By default, these are the same as the model
        project.

    Returns
    -------
    str
        The model schema returned in JSON format.

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
    """Add additional files to the model.

    Parameters
    ----------
    model : str or dict
        The name or id of the model, or a dictionary representation of the model.
    file : str or bytes
        A file related to the model, such as the model code.
    name : str
        Name of the file related to the model.
    role : str
        Role of the model file, such as 'Python pickle'.


    Returns
    -------
    str
        The model content schema.

    """

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
    """Create a model project in the given model repository.

    Parameters
    ----------
    project : str or dict
        The name or id of the model project, or a dictionary representation of the project.
    repository : str or dict
        The name or id of the model repository, or a dictionary representation of the repository.

    Returns
    -------
    RestObj

    """


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
    """Import a model and contents as a ZIP file into a model project.

    Parameters
    ----------
    name : str or dict
        The name of the model.
    project : str or dict
        The name or id of the model project, or a dictionary representation of the project.
    description : str
        The description of the model.
    file : byte
        The ZIP file containing the model and contents.

    Returns
    -------
    ResponseObj
        The API response after importing the model.

    """
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
