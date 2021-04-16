#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for converting Python model objects into SAS-compatible formats."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from .. import pzmm


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
        # !!! as of 12/9 pzmm only supposed classification !!!

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