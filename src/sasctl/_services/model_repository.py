#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Model Repository service supports registering and managing models."""

from .service import Service
from ..core import current_session, get, delete

FUNCTIONS = {'Analytical', 'Classification', 'Clustering', 'Forecasting',
             'Prediction', 'Text categorization', 'Text extraction',
             'Text sentiment', 'Text topics', 'Sentiment'}


def _get_filter(x):
    # Model Repository filtering is done using the properties= query parameter
    # instead of the default filter= parameter (as of Viya 3.4).
    # Define a custom function for building out the filter
    return dict(properties='(name, %s)' % x)

class ModelRepository(Service):
    """Implements the Model Repository REST API.

    The Model Repository API provides support for registering, organizing,
    and managing models within a common model repository.

    See Also
    --------
    `REST Documentation <https://developer.sas.com/apis/rest/DecisionManagement
    /#model-repository>`_

    """

    _SERVICE_ROOT = '/modelRepository'

    list_repositories, get_repository, update_repository, \
        delete_repository = Service._crud_funcs('/repositories', 'repository',
                                                get_filter=_get_filter)

    list_projects, get_project, update_project, \
        delete_project = Service._crud_funcs('/projects', 'project',
                                             get_filter=_get_filter)

    list_models, get_model, update_model, \
        delete_model = Service._crud_funcs('/models', 'model',
                                           get_filter=_get_filter)

    @classmethod
    def get_astore(cls, model):
        """Get the ASTORE for a model registered int he model repository.

        Parameters
        ----------
        model :  str or dict
            The name or id of the model, or a dictionary representation of the
            model.

        Returns
        ----------
        binary?

        """
        # TODO: Download binary object?

        link = cls.get_model_link(model, 'analyticStore', refresh=True)

        if link is not None:
            return link.get('href')

    @classmethod
    def get_model_link(cls, model, rel, refresh=False):
        """Retrieve a link from a model's set of links.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the
            model.
        rel : str
            The name of the link to retrieve
        refresh : bool, optional
            Whether the model's data should be refreshed before searching for
            the link.

        Returns
        -------
        dict or None
            Dictionary containing the link's properties

        """
        # Try to find the link if the model is a dictionary
        if isinstance(model, dict):
            link = cls.get_link(model, rel)
            if link is not None or not refresh:
                return link

        # Model either wasn't a dictionary or it didn't include the link
        # Pull the model data and then check for the link
        model = cls.get_model(model, refresh=refresh)

        return cls.get_link(model, rel)

    @classmethod
    def get_score_code(cls, model):
        """Get the score code for a model registered in the model repository.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model.

        Returns
        -------
        str
            The code designated as the model's score code

        """
        link = cls.get_model_link(model, 'scoreCode', refresh=True)

        if link is not None:
            scorecode_uri = link.get('href')
            return get(scorecode_uri,
                       headers={'Accept': 'text/vnd.sas.models.score.code.ds2package'})

    @classmethod
    def get_model_contents(cls, model):
        """Retrieve the additional files and data associated with the model.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the
            model.

        Returns
        -------
        list
            The code designated as the model's score code

        """
        link = cls.get_model_link(model, 'contents', refresh=True)

        return cls.request_link(link, 'contents')

    @classmethod
    def create_model(cls, model, project,
                     description=None,
                     modeler=None,
                     function=None,
                     algorithm=None,
                     tool=None,
                     score_code_type=None,
                     training_table=None,
                     event_prob_variable=None,
                     event_target_value=None,
                     is_champion=False,
                     is_challenger=False,
                     location=None,
                     target_variable=None,
                     is_retrainable=False,
                     is_immutable=False,
                     properties=None,
                     input_variables=None,
                     output_variables=None):
        """Create a model in an existing project or folder.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of the
            model.
        project : str or dict
            The name or id of the model project, or a dictionary representation
            of the model project.
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
        score_code_type : str, optional
            The score code type for the model.
        training_table : str, optional
            The train data table.
        event_prob_variable : str, optional
            The name of the event probability variable.  Used for
            classification models only.
        event_target_value : str, optional
            The target event value.  Used for classification models only.
        is_champion : bool, optional
            Indicates whether the model should be designated as the
            project's champion model.  Defaults to False.
        is_challenger : bool, optional
            Indicates whether the model should be designated as a
            challenger model in the project.  Defaults to False.
        location : str, optional,
            The location of this model.
        target_variable : str, optional
            The name of the target variable.
        is_retrainable : bool
            Indicates whether the model can be retrained or not.
        is_immutable : bool
            Indicates whether the model can be changed or not.
        properties : array_like, optional (custom properties)
            Custom model properties that can be set: name, value, type
        input_variables : array_like, optional
            Model input variables. By default, these are the same as the model
            project.
        output_variables : array_like, optional
            Model output variables. By default, these are the same as the model
             project.

        Returns
        -------
        str
            The model schema returned in JSON format.

        """
        properties = properties or {}

        if isinstance(model, str):
            model = {'name': model}
        assert isinstance(model, dict)

        p = cls.get_project(project)
        if p is None:
            raise ValueError("Unable to find project '%s'" % project)

        # Use any explicitly passed parameter value first.
        # Fall back to values in the model dict.
        model['projectId'] = p['id']
        model['modeler'] = \
            modeler or model.get('modeler') or current_session().username
        model['description'] = description or model.get('description')
        model['function'] = function or model.get('function')
        model['algorithm'] = algorithm or model.get('algorithm')
        model['tool'] = tool or model.get('tool')
        model['champion'] = is_champion or model.get('champion')

        if is_champion:
            model['role'] = 'champion'
        elif is_challenger:
            model['role'] = 'challenger'

        model.setdefault('properties', [{'name': k, 'value': v} for k, v in
                                        properties.items()])
        model['scoreCodeType'] = score_code_type or model.get('scoreCodeType')
        model['trainTable'] = training_table or model.get('trainTable')
        model['classificationEventProbabilityVariableName'] = \
            event_prob_variable \
            or model.get('classificationEventProbabilityVariableName')
        model['classificationTargetEventValue'] = \
            event_target_value or model.get('classificationTargetEventValue')
        model['location'] = location or model.get('location')
        model['targetVariable'] = \
            target_variable or model.get('targetVariable')
        model['retrainable'] = is_retrainable or model.get('retrainable')
        model['immutable'] = is_immutable or model.get('immutable')
        model['inputVariables'] = \
            input_variables or model.get('inputVariables', [])
        model['outputVariables'] = \
            output_variables or model.get('outputVariables', [])
        model['version'] = '2'

        return cls.post('/models', json=model, headers={
            'Content-Type': 'application/vnd.sas.models.model+json'})

    @classmethod
    def add_model_content(cls, model, file, name=None, role=None):
        """Add additional files to the model.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model.
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
        if cls.is_uuid(model):
            id = model
        elif isinstance(model, dict) and 'id' in model:
            id = model['id']
        else:
            model = cls.get_model(model)
            id = model['id']

        metadata = {'role': role}
        if name is not None:
            metadata['name'] = name

        return cls.post('/models/{}/contents'.format(id),
                        files={name: file}, data=metadata)

    @classmethod
    def default_repository(cls):
        """Get the built-in default repository.

        Returns
        -------
        RestObj

        """
        repo = cls.get_repository('Repository 1')  # Default in 19w04
        if repo is None:
            repo = cls.get_repository('Public')  # Default in 19w21
        if repo is None:
            all_repos = cls.list_repositories()
            if len(all_repos) > 0:
                repo = cls.list_repositories()[0]

        return repo

    @classmethod
    def create_project(cls, project, repository, **kwargs):
        """Create a model project in the given model repository.

        Parameters
        ----------
        project : str or dict
            The name or id of the model project, or a dictionary representation
            of the project.
        repository : str or dict
            The name or id of the model repository, or a dictionary
            representation of the repository.

        Returns
        -------
        RestObj

        """
        if isinstance(project, str):
            project = {'name': project}

        repository = cls.get_repository(repository)

        project['repositoryId'] = repository['id']
        project['folderId'] = repository['folderId']

        project.update(kwargs)
        return cls.post('/projects', json=project,
                        headers={
                            'Content-Type': 'application/vnd.sas.models.project+json'})

    @classmethod
    def import_model_from_zip(cls, name, project, file, description=None,
                              version='latest'):
        """Import a model and contents as a ZIP file into a model project.

        Parameters
        ----------
        name : str or dict
            The name of the model.
        project : str or dict
            The name or id of the model project, or a dictionary
            representation of the project.
        description : str
            The description of the model.
        file : bytes
            The ZIP file containing the model and contents.

        Returns
        -------
        RestObj
            The API response after importing the model.

        """
        project = cls.get_project(project)

        if project is None:
            raise ValueError('Project `%s` could not be found.' % str(project))

        params = {'name': name,
                  'description': description,
                  'type': 'ZIP',
                  'projectId': project.id,
                  'versionOption': version}
        params = '&'.join(['{}={}'.format(k, v) for k, v in params.items()])

        r = cls.post('/models#octetStream',
                     data=file.read(),
                     params=params,
                     headers={'Content-Type': 'application/octet-stream'})
        return r

    @classmethod
    def create_model_version(cls, model, minor=False):
        """Create a new version of an existing model.

        Create a new major (X.0) or minor (1.X) version of an existing
        model.  All contents from the current version are copied to the new
        version.

        Parameters
        ----------
        model : str or dict
            The name, id, or dictionary representation of a model.
        minor : bool, optional
            Whether the new version should be a minor version increment from
            the current version.  Otherwise, a new major version will be
            created.  Defaults to False.

        Returns
        -------
        RestObj
            The new version of the model.

        """
        model_obj = cls.get_model(model)
        if cls.get_model_link(model_obj, 'addModelVersion') is None:
            raise ValueError("Unable to create a new version for model '%s'"
                             % model)

        option = 'minor' if minor else 'major'

        # As of 8/9/19 addModelVersion returns the CURRENT model version,
        # not the NEW version.  Refresh the model to get the latest version.
        cls.request_link(model_obj, 'addModelVersion', json={'option': option})

        return cls.get_model(model_obj, refresh=True)

    @classmethod
    def list_model_versions(cls, model):
        """Get a list of previous versions of a model.

        The current model version is not included in the results.

        Parameters
        ----------
        model : str or dict
            The name, id, or dictionary representation of a model.


        Returns
        -------
        list

        """
        model = cls.get_model(model)
        if cls.get_model_link(model, 'modelVersions') is None:
            raise ValueError("Unable to retrieve versions for model '%s'"
                             % model)

        return cls.request_link(model, 'modelVersions')

    @classmethod
    def copy_analytic_store(cls, model):
        """Copy model ASTOREs to a pre-defined server location.

        Copies all of the analytic stores for a model to the pre-defined
        server location (/config/data/modelsvr/astore).
        To enable publishing a scoring, models that contain analytic stores
        need the ASTORE files to be copied to a set location
        (/config/data/modelsrv/astore).  This location is used for
        integration with Event Stream Processing and others. This request
        invokes an asynchronous call to copy the analytic store files. Check
        the individual analytic store uris to get the completion state:
        pending, copying, success, failure. Please review the full Model
        Manager documentation before using.

        Parameters
        ----------
        model : str or dict
            The name, id, or dictionary representation of a model.

        Returns
        -------
        RestObj

        """
        rel = 'copyAnalyticStore'

        model = cls.get_model(model)

        if cls.get_link(model, rel) is None:
            model = cls.get_model(model, refresh=True)

        return cls.request_link(model, 'copyAnalyticStore')


    @classmethod
    def delete_model_contents(cls, model):
        """Deletes all contents (files) in the model.

        Parameters
        ----------
        model : str or dict
            The name, id, or dictionary representation of a model.

        Returns
        -------

        """
        rel = 'delete'

        filelist=cls.get_model_contents(model)
        for delfile in filelist:
            modelfileuri=cls.get_link(delfile, rel)
            delete(modelfileuri['uri'])
