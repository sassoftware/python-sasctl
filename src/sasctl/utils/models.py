#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for converting Python model objects into SAS-compatible formats."""

import sys
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from .. import pzmm
from .decorators import versionadded
from .pymas.python import ds2_variables


# As of Viya 3.4 model registration fails if character fields are longer
# than 1024 characters
_DESC_MAXLEN = 1024

# As of Viya 3.4 model registration fails if user-defined properties are
# longer than 512 characters.
_PROP_VALUE_MAXLEN = 512
_PROP_NAME_MAXLEN = 60


def _property(k, v):
    return {'name': str(k)[:_PROP_NAME_MAXLEN],
            'value': str(v)[:_PROP_VALUE_MAXLEN]}


def _sklearn_to_dict(model):
    # Convert Scikit-learn values to built-in Model Manager values
    mappings = {'LogisticRegression': 'Logistic regression',
                'LinearRegression': 'Linear regression',
                'SVC': 'Support vector machine',
                'GradientBoostingClassifier': 'Gradient boosting',
                'GradientBoostingRegressor': 'Gradient boosting',
                'XGBClassifier': 'Gradient boosting',
                'XGBRegressor': 'Gradient boosting',
                'RandomForestClassifier': 'Forest',
                'DecisionTreeClassifier': 'Decision tree',
                'DecisionTreeRegressor': 'Decision tree',
                'classifier': 'classification',
                'regressor': 'prediction'}

    if hasattr(model, '_final_estimator'):
        estimator = model._final_estimator
    else:
        estimator = model
    estimator = type(estimator).__name__

    # Standardize algorithm names
    algorithm = mappings.get(estimator, estimator)

    # Standardize regression/classification terms
    analytic_function = mappings.get(model._estimator_type, model._estimator_type)

    if analytic_function == 'classification' and 'logistic' in algorithm.lower():
        target_level = 'Binary'
    elif analytic_function == 'prediction' and ('regressor' in estimator.lower() or 'regression' in algorithm.lower()):
        target_level = 'Interval'
    else:
        target_level = None

    # Can tell if multi-class .multi_class
    result = dict(
        description=str(model)[:_DESC_MAXLEN],
        algorithm=algorithm,
        scoreCodeType='ds2MultiType',
        trainCodeType='Python',
        targetLevel=target_level,
        function=analytic_function,
        tool='Python %s.%s'
             % (sys.version_info.major, sys.version_info.minor),
        properties=[_property(k, v) for k, v in model.get_params().items()]
    )

    return result


@versionadded(version='1.6')
class ModelInfo:
    """Performs static & dynamic analysis of a model to capture its properties.

    Parameters
    ----------
    model : object
        Instance of a Python model to inspect.

    """
    def __init__(self, model):

        self.instance = model
        self.name = None
        self.input_variables = {}
        self.output_variables = {}
        self.properties = []
        self.sample_input = None
        self.sample_output = None

        # Most models take DataFrame or array as input instead of individual variables.
        self.array_input = True

        if hasattr(model, '_estimator_type') and hasattr(model, 'get_params'):
            self.__init_scikit__(model)

    def set_variables(self, X, y=None, func_names=None):
        """Parse model input/output data and store variable information.

        Parameters
        ----------
        X : DataFrame
        y : DataFrame or Series
        func_names : str or [str]
            Only set the variable information for the specified function(s)

        """
        if hasattr(X, 'head'):
            self.sample_input = X.head(2)

        func_names = func_names or self.function_names
        if isinstance(func_names, str):
            func_names = [func_names]

        for name in func_names:
            self.input_variables[name] = self.parse_variable_types(X)
            # self.input_variables[name] = ds2_variables(X)

            # if y is str or list of str, use parse_arguments
            if y is None:
                continue

            if name == 'predict_proba':
                # Generate names for probability output columns
                out_names = ['P_%s' % c for c in self.class_names]

                # Compute sample output since `y` is assumed to be output for .predict()
                out_values = self.instance.predict_proba(self.sample_input)

                variables = self.parse_variable_types(out_values)
                self.output_variables[name] = OrderedDict(zip(out_names, variables.values()))
            else:
                self.output_variables[name] = self.parse_variable_types(y)


    @property
    def is_binary_classification(self):
        return len(self.class_names) == 2

    @property
    def is_multiclass_classification(self):
        return len(self.class_names) > 2

    @property
    def is_regression(self):
        return self.class_names is None or len(self.class_names) == 0

    def __init_scikit__(self, model):
        # Convert Scikit-learn values to built-in Model Manager values
        algorithms = {'LogisticRegression': 'Logistic regression',
                      'LinearRegression': 'Linear regression',
                      'SVC': 'Support vector machine',
                      'GradientBoostingClassifier': 'Gradient boosting',
                      'GradientBoostingRegressor': 'Gradient boosting',
                      'XGBClassifier': 'Gradient boosting',
                      'XGBRegressor': 'Gradient boosting',
                      'RandomForestClassifier': 'Forest',
                      'DecisionTreeClassifier': 'Decision tree',
                      'DecisionTreeRegressor': 'Decision tree'}

        analytic_functions = {'classifier': 'classification',
                              'regressor': 'prediction'}

        self.tool = 'scikit'
        self.description = str(model)
        self.class_names = None
        self.target_level = None
        self.function_names = ['predict']

        # Get the last estimator in the pipeline, if it's a pipeline.
        estimator = getattr(model, '_final_estimator', model)
        estimator = type(estimator).__name__

        # Standardize algorithm names
        self.algorithm = algorithms.get(estimator, estimator)

        # Standardize regression/classification terms
        self.function = analytic_functions.get(model._estimator_type, model._estimator_type)

        # Additional info for classification models
        if self.function == 'classification':
            self.class_names = list(model.classes_)
            self.function_names.append('predict_proba')

    def to_dict(self):
        """Convert to a dictionary that can be passed to MM create_model()"""

        info = {
            'name': self.name,
            'description': self.description,
            'function': self.function,
            'algorithm': self.algorithm,
            'tool': self.tool,
            'scoreCodeType': 'Python',
            'trainCodeType': 'Python',
            'training_table': None,
            'event_prob_variable': None,
            'event_target_value': None,
            'target_variable': None,
            'properties': self.properties,
            'input_variables': None,
            'output_variables': None
        }

        return info

    @classmethod
    def parse_variable_types(cls, data):
        """Determine model input/output data names & types.

        Parameters
        ----------
        data : any

        Returns
        -------
        OrderedDict
            Ordered dictionary of name:type pairs.
            Note: "type" is the string name of the type.

        """
        if isinstance(data, dict):
            types = OrderedDict(data)
        if hasattr(data, 'columns') and hasattr(data, 'dtypes'):
            # Pandas DataFrame or similar
            types = OrderedDict((c, data.dtypes[c].name) for c in data.columns)
        elif hasattr(data, 'name') and hasattr(data, 'dtypes'):
            # Pandas Series or similar
            types = OrderedDict([(data.name, data.dtype.name)])
        elif hasattr(data, 'ndim') and hasattr(data, 'dtype'):
            # NDArray or similar
            # No column names, but we can at least create dummy vars of the correct type
            if data.ndim > 1:
                # For a 2+D array, variables = columns
                types = OrderedDict([('var{}'.format(i), data.dtype.name) for i in range(data.shape[1])])
            else:
                # For a 1D array, can't use size of array but doesn't matter since there's always only 1 variable.
                types = OrderedDict([('var0', data.dtype.name)])
        else:
            raise RuntimeError(
                "Unable to determine input/ouput types using "
                "instance of type '%s'." % type(data)
            )

        return types


def create_package(model, name, inputs, target, train=None, valid=None, test=None):
    """
    model : any
        Python object
    name : str
        Name of model
    inputs : list of str
    target : str
    train : pandas.DataFrame
    valid : pandas.DataFrame
    test : pandas.DataFrame
    record_packages : bool

    Returns
    -------
    io.BytesIO
        zipfile

    """

    # Save actual model instance
    model_obj = model

    # Extract model properties
    model = _sklearn_to_dict(model_obj)
    model['name'] = name
    # TODO: scorecode gets set to ds2MultiType

    # Get the first data set that was provided
    data = next((df for df in [train, valid, test] if df is not None), None)

    # PZMM currently writes out files for a .zip import
    # Use a temporary directory so the user doesn't have to manage the files.
    with TemporaryDirectory() as temp_dir:
        j = pzmm.JSONFiles()

        # TODO: for classification only
        # !!! as of 12/9 pzmm only supports classification !!!

        prediction_threshold = 0.5

        # Creates fileMetadata.json
        j.writeFileMetadataJSON(name, jPath=temp_dir)

        # Creates ModelProperties.json
        j.writeModelPropertiesJSON(modelName=name,
                                      modelDesc='',
                                      targetVariable=target,
                                      modelType=model['algorithm'],
                                      modelPredictors=inputs,
                                      targetEvent=1,
                                      numTargetCategories=1,
                                      eventProbVar='EM_EVENTPROBABILITY',
                                      jPath=temp_dir)

        # creates outputVar.json
        output_vars = pd.DataFrame(columns=['EM_EVENTPROBABILITY', 'EM_CLASSIFICATION'])
        output_vars['EM_CLASSIFICATION'] = data[target].astype('category').cat.categories.astype('str')
        output_vars['EM_EVENTPROBABILITY'] = prediction_threshold  # Event threshold
        j.writeVarJSON(output_vars, isInput=False, jPath=temp_dir)

        # creates inputVar.json
        j.writeVarJSON(data[inputs], isInput=True, jPath=temp_dir)

        # Given a folder, write out the model to an arbitrary pickle file
        # Shouldn't this return the filename or something?
        # How to determine where it was written?
        pzmm.PickleModel.pickleTrainedModel(model, 'prefix', Path(temp_dir))

        # creates "MyRandomForestModelScore.py"
        scorecode = pzmm.ScoreCode()
        scorecode.writeScoreCode(data[inputs], data[target], name, '{}.predict({})', name + '.pickle',
                                 threshPrediction=prediction_threshold,
                                 pyPath=temp_dir)

        zipfile = pzmm.ZipModel.zipFiles(temp_dir, name)

    return zipfile