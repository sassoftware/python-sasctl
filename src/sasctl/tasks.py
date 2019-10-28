#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Commonly used tasks in the analytics life cycle."""

import json
import logging
import math
import pickle
import os
import re
import sys
import warnings

from six.moves.urllib.error import HTTPError

from . import utils
from .core import RestObj, current_session, get, get_link, request_link
from .exceptions import AuthorizationError
from .services import model_management as mm
from .services import model_publish as mp
from .services import model_repository as mr
from .utils.pymas import PyMAS, from_pickle
from .utils.misc import installed_packages


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
                'classifier': 'classification',
                'regressor': 'prediction'}

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
                   version=None, files=None, force=False):
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
    input : DataFrame, type, list of type, or dict of str: type, optional
        The expected type for each input value of the target function.
        Can be omitted if target function includes type hints.  If a DataFrame
        is provided, the columns will be inspected to determine type information.
        If a single type is provided, all columns will be assumed to be that type,
        otherwise a list of column types or a dictionary of column_name: type
        may be provided.
    version : {'new', 'latest', int}, optional
        Version number of the project in which the model should be created.
        Defaults to 'new'.
    files : list
        A list of dictionaries of the form {'name': filename, 'file': filecontent}.
        An optional 'role' key is supported for designating a file as score code,
        astore, etc.
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

    .. versionchanged:: v1.3
        Create requirements.txt with installed packages.

    """
    # TODO: Create new version if model already exists

    # If version not specified, default to creating a new version
    version = version or 'new'

    files = files or []

    # Find the project if it already exists
    p = mr.get_project(project) if project is not None else None

    # Do we need to create the project first?
    create_project = True if p is None and force else False

    if p is None and not create_project:
        raise ValueError("Project '{}' not found".format(project))

    # Use default repository if not specified
    try:
        if repository is None:
            repo_obj = mr.default_repository()
        else:
            repo_obj = mr.get_repository(repository)
    except HTTPError as e:
        if e.code == 403:
            raise AuthorizationError('Unable to register model.  User account '
                                     'does not have read permissions for the '
                                     '/modelRepository/repositories/ URL. '
                                     'Please contact your SAS Viya '
                                     'administrator.')
        else:
            raise e

    # Unable to find or create the repo.
    if repo_obj is None and repository is None:
        raise ValueError("Unable to find a default repository")
    elif repo_obj is None:
        raise ValueError("Unable to find repository '{}'".format(repository))

    # If model is a CASTable then assume it holds an ASTORE model.
    # Import these via a ZIP file.
    if 'swat.cas.table.CASTable' in str(type(model)):
        zipfile = utils.create_package(model, input=input)

        if create_project:
            outvar=[]
            invar=[]
            import zipfile as zp
            import copy
            zipfilecopy = copy.deepcopy(zipfile)
            tmpzip=zp.ZipFile(zipfilecopy)
            if "outputVar.json" in tmpzip.namelist():
                outvar=json.loads(tmpzip.read("outputVar.json").decode('utf=8')) #added decode for 3.5 and older
                for tmp in outvar:
                    tmp.update({'role':'output'})
            if "inputVar.json" in tmpzip.namelist():
                invar=json.loads(tmpzip.read("inputVar.json").decode('utf-8')) #added decode for 3.5 and older
                for tmp in invar:
                    if tmp['role'] != 'input':
                       tmp['role']='input'
            vars=invar + outvar
            project = mr.create_project(project, repo_obj, variables=vars)

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

        # Get package versions in environment
        packages = installed_packages()
        if packages is not None:
            model.setdefault('properties', [])

            # Define a custom property to capture each package version
            for p in packages:
                n, v = p.split('==')
                model['properties'].append({
                    'name': 'env_%s' % n,
                    'value': v
                })

            # Generate and upload a requirements.txt file
            files.append({'name': 'requirements.txt',
                          'file': '\n'.join(packages)})

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
            files.append({'name': 'python_wrapper.py',
                          'file': mas_module.score_code(dest='Python')})

            model['inputVariables'] = [var.as_model_metadata()
                                       for var in mas_module.variables
                                       if not var.out]

            model['outputVariables'] = \
                [var.as_model_metadata() for var in mas_module.variables
                 if var.out and var.name not in ('rc', 'msg')]
        except ValueError:
            # PyMAS creation failed, most likely because input data wasn't
            # provided
            logger.exception('Unable to inspect model %s', model)

            warnings.warn('Unable to determine input/output variables. '
                          ' Model variables will not be specified and some '
                          'model functionality may not be available.')
    else:
        # Otherwise, the model better be a dictionary of metadata
        assert isinstance(model, dict)

    if create_project:
        vars = model.get('inputVariables', [])[:]
        vars += model.get('outputVariables', [])

        function = model.get('function', '').lower()
        algorithm = model.get('algorithm', '').lower()

        if function == 'classification' and 'logistic' in algorithm:
            target_level = 'Binary'
        elif function == 'prediction' and 'regression' in algorithm:
            target_level = 'Interval'
        else:
            target_level = None

        if len(model.get('outputVariables', [])) == 1:
            var = model['outputVariables'][0]
            prediction_variable = var['name']
        else:
            prediction_variable = None

        # As of Viya 3.4 the 'predictionVariable' parameter is not set during
        # project creation.  Update the project if necessary.
        if function == 'prediction':   #Predications require predictionVariable
            project = mr.create_project(project, repo_obj,
                                    variables=vars,
                                    function=model.get('function'),
                                    targetLevel=target_level,
                                    predictionVariable=prediction_variable)

            if project.get('predictionVariable') != prediction_variable:
                project['predictionVariable'] = prediction_variable
                mr.update_project(project)
        else:  #Classifications require eventProbabilityVariable 
            project = mr.create_project(project, repo_obj,
                                    variables=vars,
                                    function=model.get('function'),
                                    targetLevel=target_level,
                                    eventProbabilityVariable=prediction_variable)
            if project.get('eventProbabilityVariable') != prediction_variable:
                project['eventProbabilityVariable'] = prediction_variable
                mr.update_project(project)

    # If replacing an existing version, make sure the model version exists
    if str(version).lower() != 'new':
        #Update an existing model with new files
        model_obj = mr.get_model(name)
        if model_obj is None:
            raise ValueError("Unable to update version '%s' of model '%s%.  "
                             "Model not found." % (version, name))
        model = mr.create_model_version(name)
        mr.delete_model_contents(model)
    else:
        #Assume new model to create
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
    module_url = _parse_module_url(msg)
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


def update_model_performance(data, model, label, refresh=True):
    """Upload data for calculating model performance metrics.

    Model performance and data distributions can be tracked over time by
    designating one or more tables that contain input data and target values.
    Performance metrics can be updated by uploading a data set for a new time
    period and executing the performance definition.

    Parameters
    ----------
    data : Dataframe
    model : str or dict
        The name or id of the model, or a dictionary representation of
        the model.
    label : str
        The time period the data is from.  Should be unique and will be
        displayed on performance charts.  Examples: 'Q1', '2019', 'APR2019'.
    refresh : bool, optional
        Whether to execute the performance definition and refresh results with
        the new data.

    Returns
    -------
    CASTable
        The CAS table containing the performance data.

    See Also
    --------
     :meth:`model_management.create_performance_definition <.ModelManagement.create_performance_definition>`

    .. versionadded:: v1.3

    """
    from .services import model_management as mm
    try:
        import swat
    except ImportError:
        raise RuntimeError("The 'swat' package is required to save model "
                           "performance data.")

    # Default to true
    refresh = True if refresh is None else refresh

    model_obj = mr.get_model(model)

    if model_obj is None:
        raise ValueError('Model %s was not found.', model)

    project = mr.get_project(model_obj.projectId)

    if project.get('function', '').lower() not in ('prediction', 'classification'):
        raise ValueError("Performance monitoring is currently supported for "
                         "regression and binary classification projects.  "
                         "Received project with '%s' function.  Should be "
                         "'Prediction' or 'Classification'.",
                         project.get('function'))
    elif project.get('targetLevel', '').lower() not in ('interval', 'binary'):
        raise ValueError("Performance monitoring is currently supported for "
                         "regression and binary classification projects.  "
                         "Received project with '%s' target level.  Should be "
                         "'Interval' or 'Binary'.", project.get('targetLevel'))
    elif project.get('predictionVariable', '') == '' and project.get('function', '').lower() == 'prediction':
        raise ValueError("Project '%s' does not have a prediction variable "
                         "specified." % project)
    elif project.get('eventProbabilityVariable', '') == '' and project.get('function', '').lower() == 'classification':
        raise ValueError("Project '%s' does not have an Event Probability variable "
                         "specified." % project)

    # Find the performance definition for the model
    # As of Viya 3.4, no way to search by model or project
    perf_def = None
    for p in mm.list_performance_definitions():
        if model_obj.id in p.modelIds:
            perf_def = p
            break

    if perf_def is None:
        raise ValueError("Unable to find a performance definition for model "
                         "'%s'" % model)

    # Check where performance datasets should be uploaded
    cas_id = perf_def['casServerId']
    caslib = perf_def['dataLibrary']
    table_prefix = perf_def['dataPrefix']

    # All input variables must be present
    missing_cols = [col for col in perf_def.inputVariables if
                    col not in data.columns]
    if len(missing_cols):
        raise ValueError("The following columns were expected but not found in "
                         "the data set: %s" % ', '.join(missing_cols))

    # If CAS is not executing the model then the output variables must also be
    # provided
    if not perf_def.scoreExecutionRequired:
        missing_cols = [col for col in perf_def.outputVariables if
                        col not in data.columns]
        if len(missing_cols):
            raise ValueError(
                "The following columns were expected but not found in the data "
                "set: %s" % ', '.join(missing_cols))

    sess = current_session()
    url = '{}://{}/{}-http/'.format(sess._settings['protocol'],
                                    sess.hostname,
                                    cas_id)
    regex = r'{}_(\d)_.*_{}'.format(table_prefix,
                                  model_obj.id)

    # Save the current setting before overwriting
    orig_sslreqcert = os.environ.get('SSLREQCERT')

    # If SSL connections to microservices are not being verified, don't attempt
    # to verify connections to CAS - most likely certs are not in place.
    if not sess.verify:
        os.environ['SSLREQCERT'] = 'no'

    # Upload the performance data to CAS
    with swat.CAS(url,
                  username=sess.username,
                  password=sess._settings['password']) as s:

        s.setsessopt(messagelevel='warning')

        with swat.options(exception_on_severity=2):
            caslib_info = s.table.tableinfo(caslib=caslib)

        all_tables = getattr(caslib_info, 'TableInfo', None)
        if all_tables is not None:
            # Find tables with similar names
            perf_tables = all_tables.Name.str.extract(regex,
                                                      flags=re.IGNORECASE,
                                                      expand=False)

            # Get last-used sequence number
            last_seq = perf_tables.dropna().astype(int).max()
            next_seq = 1 if math.isnan(last_seq) else last_seq + 1
        else:
            next_seq = 1

        table_name = '{prefix}_{sequence}_{label}_{model}'.format(
            prefix=table_prefix,
            sequence=next_seq,
            label=label,
            model=model_obj.id
        )

        with swat.options(exception_on_severity=2):
            # Table must be promoted so performance jobs can access.
            tbl = s.upload(data, casout=dict(name=table_name,
                                                caslib=caslib,
                                                promote=True)).casTable

    # Restore the original value
    if orig_sslreqcert is not None:
        os.environ['SSLREQCERT'] = orig_sslreqcert

    # Execute the definition if requested
    if refresh:
        mm.execute_performance_definition(perf_def)

    return tbl


def _parse_module_url(msg):
    try:
        details = json.loads(msg)

        module_url = get_link(details, 'module')
        module_url = module_url.get('href')
    except json.JSONDecodeError:
        match = re.search(r'(?:rel=module, href=(.*?),)', msg)
        module_url = match.group(1) if match else None

    return module_url
