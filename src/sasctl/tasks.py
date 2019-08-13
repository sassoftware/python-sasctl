#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Commonly used tasks in the analytics life cycle."""

import json
import logging
import pickle
import re
import sys
import warnings

from . import utils
from .core import RestObj, get, get_link, request_link
from .services import model_management as mm
from .services import model_publish as mp
from .services import model_repository as mr
from .utils.pymas import PyMAS, from_pickle


logger = logging.getLogger(__name__)


def _sklearn_to_dict(model):
    # As of Viya 3.4 model registration fails if character fields are longer
    # than 1024 characters
    DESC_MAXLEN = 1024

    # As of Viya 3.4 model registration fails if user-defined properties are
    # longer than 512 characters.
    PROP_MAXLEN = 512

    # Convert Scikit-learn values to built-in Model Manager values
    mappings = {'LogisticRegression': 'Logistic regression',
                'LinearRegression': 'Linear regression',
                'SVC': 'Support vector machine',
                'GradientBoostingClassifier': 'Gradient boosting',
                'XGBClassifier': 'Gradient boosting',
                'XGBRegressor': 'Gradient boosting',
                'RandomForestClassifier': 'Forest',
                'DecisionTreeClassifier': 'Decision tree',
                'DecisionTreeRegressor': 'Decision tree',
                'classifier': 'Classification',
                'regressor': 'Prediction'}

    if hasattr(model, '_final_estimator'):
        estimator = type(model._final_estimator)
    else:
        estimator = type(model)

    # Can tell if multi-class .multi_class
    result = dict(
        description=str(model)[:DESC_MAXLEN],
        algorithm=mappings.get(estimator.__name__, estimator.__name__),
        scoreCodeType='ds2MultiType',
        trainCodeType='Python',
        function=mappings.get(model._estimator_type, model._estimator_type),
        tool='Python %s.%s'
             % (sys.version_info.major, sys.version_info.minor),
        properties=[{'name': str(k)[:PROP_MAXLEN],
                     'value': str(v)[:PROP_MAXLEN]}
                    for k, v in model.get_params().items()]
    )

    return result


def register_model(model, name, project, repository=None, input=None,
                   version='latest', files=None, force=False):
    """Register a model in the model repository.

    Parameters
    ----------
    model : swat.CASTable or sklearn.BaseEstimator
        The model to register.  If an instance of ``swat.CASTable`` the table
        is assumed to hold an ASTORE, which will be downloaded and used to
        construct the model to register.  If a scikit-learn estimator, the
        model will be pickled and uploaded to the registry and score code will
        be generated for publishing the model to MAS.
    name : str
        Designated name for the model in the repository.
    project : str or dict
        The name or id of the project, or a dictionary representation of
        the project.
    repository : str or dict, optional
        The name or id of the repository, or a dictionary representation of
        the repository.  If omitted, the default repository will be used.
    input
    version : {'new', 'latest', int}, optional
        Version number of the project in which the model should be created.
    files :
    force : bool, optional
        Create dependencies such as projects and repositories if they do not
        already exist.

    Returns
    -------
    model : RestObj
        The newly registered model as an instance of ``RestObj``

    Notes
    -----
    If the specified model is a CAS table the model data and metadata will be
    written to a temporary zip file and then imported using
    model_repository.import_model_from_zip.

    If the specified model is from the Scikit-Learn package, the model will be
    created using model_repository.create_model and any additional files will
    be uploaded as content.

    """
    # TODO: Create new version if model already exists
    # TODO: Allow file info to be specified
    # TODO: Performance stats

    files = files or []

    # Find the project if it already exists
    p = mr.get_project(project) if project is not None else None

    # Do we need to create the project first?
    create_project = True if p is None and force else False

    if p is None and not create_project:
        raise ValueError("Project '{}' not found".format(project))

    if repository is None:
        repository = mr.default_repository()
    else:
        repository = mr.get_repository(repository)

    # Unable to find or create the repo.
    if repository is None:
        raise ValueError("Unable to find repository '{}'".format(repository))

    # If model is a CASTable then assume it holds an ASTORE model.
    # Import these via a ZIP file.
    if 'swat.cas.table.CASTable' in str(type(model)):
        zipfile = utils.create_package_from_astore(model)

        if create_project:
            project = mr.create_project(project, repository)

        model = mr.import_model_from_zip(name, project, zipfile,
                                         version=version)
        return model

    # If the model is an scikit-learn model, generate the model dictionary
    # from it and pickle the model for storage
    elif all(hasattr(model, attr) for attr
             in ['_estimator_type', 'get_params']):
        # Pickle the model so we can store it
        model_pkl = pickle.dumps(model)
        files.append({'name': 'model.pkl',
                      'file': model_pkl,
                      'role': 'Python Pickle'})

        # Extract model properties
        model = _sklearn_to_dict(model)
        model['name'] = name

        # Generate PyMAS wrapper
        try:
            mas_module = from_pickle(model_pkl, 'predict',
                                     input_types=input, array_input=True)
            assert isinstance(mas_module, PyMAS)

            # Include score code files from ESP and MAS
            files.append({'name': 'dmcas_packagescorecode.sas',
                          'file': mas_module.score_code(),
                          'role': 'Score Code'})
            files.append({'name': 'dmcas_epscorecode.sas',
                          'file': mas_module.score_code(dest='CAS'),
                          'role': 'score'})

            model['inputVariables'] = [var.as_model_metadata()
                                       for var in mas_module.variables
                                       if not var.out]

            model['outputVariables'] = \
                [var.as_model_metadata() for var in mas_module.variables
                 if var.out and var.name not in ('rc', 'msg')]
        except ValueError:
            # PyMAS creation failed, most likely because input data wasn't
            # provided
            warnings.warn('Unable to determine input/output variables. '
                          ' Model variables will not be specified.')
    else:
        # Otherwise, the model better be a dictionary of metadata
        assert isinstance(model, dict)

    if create_project:
        vars = model.get('inputVariables', [])
        vars += model.get('outputVariables', [])

        if model.get('function') == 'Regression':
            target_level = 'Interval'
        else:
            target_level = None

        project = mr.create_project(project, repository,
                                    variables=vars,
                                    targetLevel=target_level)

    model = mr.create_model(model, project)

    assert isinstance(model, RestObj)

    # Upload any additional files
    for file in files:
        if isinstance(file, dict):
            mr.add_model_content(model, **file)
        else:
            mr.add_model_content(model, file)

    return model


def publish_model(model,
                  destination,
                  code=None,
                  max_retries=60,
                  replace=False, **kwargs):
    """Publish a model to a configured publishing destination.

    Parameters
    ----------
    model : str or dict
        The name or id of the model, or a dictionary representation of
        the model.
    destination : str
    code : optional
    max_retries : int, optional
    replace : bool, optional
        Whether to overwrite the model if it already exists in
        the `destination`
    kwargs : optional
        additional arguments will be passed to the underlying publish
        functions.

    Returns
    -------
    RestObj
        The published model

    Notes
    -----
    If no code is specified, the model is assumed to be already registered in
    the model repository and Model Manager's publishing functionality will be
    used.

    Otherwise, the model publishing API will be used.

    See Also
    --------
    :meth:`model_management.publish_model <.ModelManagement.publish_model>`
    :meth:`model_publish.publish_model <.ModelPublish.publish_model>`


    .. versionchanged:: 1.1.0
       Added `replace` option.

    """
    def submit_request():
        # Submit a publishing request
        if code is None:
            dest_obj = mp.get_destination(destination)

            if dest_obj and dest_obj.destinationType == "cas":
                publish_req = mm.publish_model(model, destination,
                                               force=replace,
                                               reload_model_table=True)
            else:
                publish_req = mm.publish_model(model, destination,
                                               force=replace)
        else:
            publish_req = mp.publish_model(model, destination,
                                           code=code, **kwargs)

        # A successfully submitted request doesn't mean a successfully
        # published model.  Response for publish request includes link to
        # check publish log
        job = mr._monitor_job(publish_req, max_retries=max_retries)
        return job

    # Submit and wait for status
    job = submit_request()

    # If model was successfully published and it isn't a MAS module, we're done
    if job.state.lower() == 'completed' \
            and job.destination.destinationType != 'microAnalyticService':
            return request_link(job,'self')

    # If MAS publish failed and replace=True, attempt to delete the module
    # and republish
    if job.state.lower() == 'failed' and replace and \
            job.destination.destinationType == 'microAnalyticService':
            from .services import microanalytic_score as mas
            mas.delete_module(job.publishName)

            # Resubmit the request
            job = submit_request()

    # Raise exception if still failing
    if job.state.lower() == 'failed':
        log = request_link(job, 'publishingLog')
        raise RuntimeError("Failed to publish model '%s': %s"
                           % (model, log.log))

    # Raise exception if unknown status received
    elif job.state.lower() != 'completed':
        raise RuntimeError("Model publishing job in an unknown state: '%s'"
                           % job.state.lower())

    log = request_link(job, 'publishingLog')
    msg = log.get('log').lstrip('SUCCESS===')

    # As of Viya 3.4 MAS converts module names to lower case.
    # Since we can't rely on the request module name being preserved, try to
    # parse the URL out of the response so we can retrieve the created module.
    try:
        details = json.loads(msg)

        module_url = get_link(details, 'module')
        module_url = module_url.get('href')
    except json.JSONDecodeError:
        match = re.search(r'(?:rel=module, href=(.*?),)', msg)
        module_url = match.group(1) if match else None

    if module_url is None:
        raise Exception('Unable to retrieve module URL from publish log.')

    module = get(module_url)

    if 'application/vnd.sas.microanalytic.module' in module._headers[
        'content-type']:
        # Bind Python methods to the module instance that will execute the
        # corresponding MAS module step.
        from sasctl.services import microanalytic_score as mas
        return mas.define_steps(module)
    return module
