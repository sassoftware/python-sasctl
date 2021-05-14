#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculates and formats model statistics for inclusion in SAS Model Manager."""

import logging

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

from math import sqrt

try:
    import sklearn
    from sklearn.metrics import roc_curve, log_loss, roc_auc_score
    from sklearn.metrics import accuracy_score, mean_squared_error
except ImportError:
    sklearn = None

log = logging.getLogger(__name__)


def lift_statistics(model, train=None, valid=None, test=None, event=None):
    """Calculate lift statistics for a model.

    Parameters
    ----------
    model : predictor-like
        An object that conforms to scikit-learn's Predictor interface and
        implements a `predict_proba` method.
    train : (array_like, array_like), optional
        A tuple of the training inputs and target output.
    valid : (array_like, array_like), optional
        A tuple of the validation inputs and target output.
    test : (array_like, array_like), optional
        A tuple of the test inputs and target output.
    event : str or int, optional
        The model output value that corresponds to a positive event in a
        binary classification task.

    Returns
    -------
    dict
        Metrics calculated for each dataset and formatted as expected by SAS
        Model Manager.

    """
    datasets = (valid, train, test)
    labels = ['VALIDATE', 'TRAIN', 'TEST']

    if not hasattr(model, 'classes_'):
        return {}

    # At least some combination of datasets must be provided
    if all(d is None for d in datasets):
        raise ValueError("At least one dataset must be provided.")

    results = []
    row_count = 0
    class_names = list(model.classes_)
    num_classes = len(class_names)

    if event is None:
        event = num_classes - 1
    elif event in class_names:
        event = class_names.index(event)
    else:
        event = int(event)

    for idx, dataset in enumerate(datasets):
        if dataset is None:
            continue

        X, y_true = dataset

        target_column = getattr(y_true, 'name', 'Class')
        proba_columns = ['P_%s%d' % (target_column, i) for i in range(num_classes)]
        event_column = proba_columns[event]

        # We need to re-assign int values to the dataset to ensure they match
        # up to the column order output by the model.  Otherwise, output labels
        # will match, but underlying int representations will be off.
        y_true = y_true.astype('category').cat.reorder_categories(model.classes_)

        # Predicted probability for each class
        y_pred_probs = pd.DataFrame(model.predict_proba(X))

        # Maximum likelihood class for each observation
        y_pred_index = y_pred_probs.idxmax(axis=1)

        # Column names default to 0 / 1.  Only rename after calculating idxmax
        # to ensure y_pred_index contains 0 / 1 values.
        y_pred_probs.columns = proba_columns

        # Explicitly reset indexes.  pd.concat(ignore_index=True) didn't work.
        y_pred_probs.reset_index(drop=True, inplace=True)
        y_pred_index.reset_index(drop=True, inplace=True)
        y_true_codes = y_true.cat.codes
        y_true_codes.reset_index(drop=True, inplace=True)

        df = pd.concat((y_pred_probs, y_pred_index, y_true_codes), axis=1)
        df.columns = proba_columns + ['predicted', 'target']

        # Sort by highest probability of event (according to model)
        df.sort_values(by=event_column, ascending=False, inplace=True)

        num_groups = 20
        df['group'] = pd.cut(np.arange(len(df)), num_groups).codes

        total_samples = len(df)

        actualValue = df['target']
        predictValue = df['predicted']      # TODO: Use actual or predicted events?
        group_ids = df['group']

        # Count of each label by group
        label_counts_by_group = pd.crosstab(group_ids, actualValue)

        # Count of event label by group
        event_counts_by_group = label_counts_by_group[event]

        # Cumulative total of event label by group
        cum_events_by_group = event_counts_by_group.cumsum()

        # Total # of observations in each group
        obs_counts_by_group = label_counts_by_group.sum(axis=1)

        # Cumulative total of observations by group
        cum_obs_by_group = obs_counts_by_group.cumsum()

        # Percent of total observations in each group
        obs_percent_by_group = 100 * obs_counts_by_group / total_samples

        # Number of events in full dataset
        total_event_count = event_counts_by_group.sum()

        # Percent of total events included in group
        gain_by_group = 100 * event_counts_by_group / total_event_count

        # Percent of obs in group that are event
        event_percent_by_group = 100 * event_counts_by_group / obs_counts_by_group

        total_response_percent = 100 * (total_event_count / total_samples)
        lift = event_percent_by_group / total_response_percent
        cum_response_percent = 100 * cum_events_by_group / cum_obs_by_group

        df = pd.DataFrame({'Quantile Number': obs_counts_by_group,
                           'Quantile Percent': obs_percent_by_group,
                           'Gain Number': event_counts_by_group,
                           'Gain Percent': gain_by_group,
                           'Response Percent': event_percent_by_group,
                           'Lift': lift,
                           'Acc Quantile Number': cum_obs_by_group,
                           'Acc Quantile Percent': 100 * cum_obs_by_group / total_samples,  # Cumulate % of obs by group
                           'Acc Gain Number': cum_events_by_group,
                           'Acc Gain Percent': 100 * cum_events_by_group / total_event_count,
                           'Acc Response Percent': cum_response_percent,
                           'Acc Lift': cum_response_percent / total_response_percent
                           })

        for index, row in df.iterrows():
            row_count += 1

            stats = {'header': None,
                     'rowNumber': row_count,
                     'dataMap': {'_DataRole_': labels[idx],
                                 '_Column_': target_column,
                                 '_Event_': event,
                                 '_Depth_': row['Acc Quantile Percent'],
                                 '_NObs_': row['Quantile Number'],
                                 '_Gain_': row['Gain Number'],
                                 '_Resp_': row['Gain Percent'],
                                 '_CumResp_': row['Acc Gain Percent'],
                                 '_PctResp_': row['Response Percent'],
                                 '_CumPctResp_': row['Acc Response Percent'],
                                 '_Lift_': row['Lift'],
                                 '_CumLift_': row['Acc Lift']
                                 }}

            results.append(stats)

    return {'name': 'dmcas_lift',
            'parameterMap': {
                '_DataRole_': {
                    'parameter': '_DataRole_',
                    'type': 'char',
                    'label': 'Data Role',
                    'length': 10,
                    'order': 1,
                    'values': ['_DataRole_'],
                    'preformatted': False},
                '_PartInd_': {
                    'parameter': '_PartInd_',
                    'type': 'num',
                    'label': 'Partition Indicator',
                    'length': 8,
                    'order': 2,
                    'values': ['_PartInd_'],
                    'preformatted': False},
                '_PartInd__f': {
                    'parameter': '_PartInd__f',
                    'type': 'char',
                    'label': 'Formatted Partition',
                    'length': 12,
                    'order': 3,
                    'values': ['_PartInd__f'],
                    'preformatted': False},
                '_Column_': {
                    'parameter': '_Column_',
                    'type': 'char',
                    'label': 'Analysis Variable',
                    'length': 32,
                    'order': 4,
                    'values': ['_Column_'],
                    'preformatted': False},
                '_Event_': {
                    'parameter': '_Event_',
                    'type': 'char',
                    'label': 'Event',
                    'length': 8,
                    'order': 5,
                    'values': ['_Event_'],
                    'preformatted': False},
                '_Depth_': {
                    'parameter': '_Depth_',
                    'type': 'num',
                    'label': 'Depth',
                    'length': 8,
                    'order': 7,
                    'values': ['_Depth_'],
                    'preformatted': False},
                '_NObs_': {
                    'parameter': '_NObs_',
                    'type': 'num',
                    'label': 'Sum of Frequencies',
                    'length': 8,
                    'order': 8,
                    'values': ['_NObs_'],
                    'preformatted': False},
                '_Gain_': {
                    'parameter': '_Gain_',
                    'type': 'num',
                    'label': 'Gain',
                    'length': 8,
                    'order': 9,
                    'values': ['_Gain_'],
                    'preformatted': False},
                '_Resp_': {
                    'parameter': '_Resp_',
                    'type': 'num',
                    'label': '% Captured Response',
                    'length': 8,
                    'order': 10,
                    'values': ['_Resp_'],
                    'preformatted': False},
                '_CumResp_': {
                    'parameter': '_CumResp_',
                    'type': 'num',
                    'label': 'Cumulative % Captured Response',
                    'length': 8,
                    'order': 11,
                    'values': ['_CumResp_'],
                    'preformatted': False},
                '_PctResp_': {
                    'parameter': '_PctResp_',
                    'type': 'num',
                    'label': '% Response',
                    'length': 8,
                    'order': 12,
                    'values': ['_PctResp_'],
                    'preformatted': False},
                '_CumPctResp_': {
                    'parameter': '_CumPctResp_',
                    'type': 'num',
                    'label': 'Cumulative % Response',
                    'length': 8,
                    'order': 13,
                    'values': ['_CumPctResp_'],
                    'preformatted': False},
                '_Lift_': {
                    'parameter': '_Lift_',
                    'type': 'num',
                    'label': 'Lift',
                    'length': 8,
                    'order': 14,
                    'values': ['_Lift_'],
                    'preformatted': False},
                '_CumLift_': {
                    'parameter': '_CumLift_',
                    'type': 'num',
                    'label': 'Cumulative Lift',
                    'length': 8,
                    'order': 15,
                    'values': ['_CumLift_'],
                    'preformatted': False}},
            'data': results,
            'version': 1}


def roc_statistics(model, train=None, valid=None, test=None):
    """

    Parameters
    ----------
    model : predictor-like
        An object that conforms to scikit-learn's Predictor interface and
        implements a `predict` method.
    train : (array_like, array_like)
        A tuple of the training inputs and target output.
    valid : (array_like, array_like)
        A tuple of the validation inputs and target output.
    test : (array_like, array_like)
        A tuple of the test inputs and target output.

    Returns
    -------
    dict
        Metrics calculated for each dataset and formatted as expected by SAS
        Model Manager.

    """
    datasets = (valid, train, test)
    labels = ['VALIDATE', 'TRAIN', 'TEST']

    # At least some combination of datasets must be provided
    if all(d is None for d in datasets):
        raise ValueError("At least one dataset must be provided.")

    results = []
    row_count = 0

    for idx, dataset in enumerate(datasets):
        if dataset is None or not hasattr(model, 'classes_') or len(model.classes_ != 2): # multiclass not supported
            continue

        X, y_true = dataset

        y_pred = model.predict(X)
        y_pred = pd.Series(y_pred, dtype='category')

        fpr, tpr, threshold = roc_curve(y_true.astype('category').cat.codes, y_pred.cat.codes)

        for f, t, h in zip(fpr, tpr, threshold):
            row_count += 1
            stats = {'rowNumber': row_count,
                     'header': None,
                     'dataMap': {
                         '_DataRole_': labels[idx],
                         '_PartInd_': str(idx),
                         '_PartInd__f': '           %d' % idx,
                         '_Sensitivity_': t,
                         '_Specificity_': (1 - f),
                         '_OneMinusSpecificity_': 1 - (1 - f),
                         '_Event_': 1,
                         '_Cutoff_': int(h),
                         '_FPR_': f}
                     }
            results.append(stats)

        return {'name': 'dmcas_roc',
                'revision': 0,
                'order': 0,
                'type': None,
                'parameterMap': {
                    '_DataRole_': {
                        'parameter': '_DataRole_',
                        'type': 'char',
                        'label': 'Data Role',
                        'length': 10,
                        'order': 1,
                        'values': ['_DataRole_'],
                        'preformatted': False},
                    '_PartInd_': {
                        'parameter': '_PartInd_',
                        'type': 'num',
                        'label': 'Partition Indicator',
                        'length': 8,
                        'order': 2,
                        'values': ['_PartInd_'],
                        'preformatted': False},
                    '_PartInd__f': {
                        'parameter': '_PartInd__f',
                        'type': 'char',
                        'label': 'Formatted Partition',
                        'length': 12,
                        'order': 3,
                        'values': ['_PartInd__f'],
                        'preformatted': False},
                    '_Column_': {
                        'parameter': '_Column_',
                        'type': 'num',
                        'label': 'Analysis Variable',
                        'length': 32,
                        'order': 4,
                        'values': ['_Column_'],
                        'preformatted': False},
                    '_Event_': {
                        'parameter': '_Event_',
                        'type': 'char',
                        'label': 'Event',
                        'length': 8,
                        'order': 5,
                        'values': ['_Event_'],
                        'preformatted': False},
                    '_Cutoff_': {
                        'parameter': '_Cutoff_',
                        'type': 'num',
                        'label': 'Cutoff',
                        'length': 8,
                        'order': 6,
                        'values': ['_Cutoff_'],
                        'preformatted': False},
                    '_Sensitivity_': {
                        'parameter': '_Sensitivity_',
                        'type': 'num',
                        'label': 'Sensitivity',
                        'length': 8,
                        'order': 7,
                        'values': ['_Sensitivity_'],
                        'preformatted': False},
                    '_Specificity_': {
                        'parameter': '_Specificity_',
                        'type': 'num',
                        'label': 'Specificity',
                        'length': 8,
                        'order': 8,
                        'values': ['_Specificity_'],
                        'preformatted': False},
                    '_FPR_': {
                        'parameter': '_FPR_',
                        'type': 'num',
                        'label': 'False Positive Rate',
                        'length': 8,
                        'order': 9,
                        'values': ['_FPR_'],
                        'preformatted': False},
                    '_OneMinusSpecificity_': {
                        'parameter': '_OneMinusSpecificity_',
                        'type': 'num',
                        'label': '1 - Specificity',
                        'length': 8,
                        'order': 10,
                        'values': ['_OneMinusSpecificity_'],
                        'preformatted': False}},
                'data': results,
                'version': 1,
                'xInteger': False,
                'yInteger': False}


def _convert_class_labels(model, values):
    """Convert classification labels to integer codes."""

    # If it's not even a classification model, don't bother
    if not hasattr(model, 'classes_'):
        return values

    if hasattr(values, 'cat'):
        # Re-assign int values to the categorical data to ensure they match up
        # with the column values output by the model.
        values = values.cat.reorder_categories(model.classes_)

        # Return integer values
        return values.cat.codes

    # Construct a new array by replacing labels with their corresponding code
    # as output by the model.
    new_values = np.zeros(len(values), dtype='int')
    for i, c in enumerate(model.classes_):
        mask = values == c
        new_values[mask] = i

    return new_values


def fit_statistics(model, train=None, valid=None, test=None, event=None):
    """Calculate model fit statistics.

    Parameters
    ----------
    model : predictor-like
        An object that conforms to scikit-learn's Predictor interface and
        implements a `predict` method.
    train : (array_like, array_like)
        A tuple of the training inputs and target output.
    valid : (array_like, array_like)
        A tuple of the validation inputs and target output.
    test : (array_like, array_like)
        A tuple of the test inputs and target output.
    event : str or int, optional
        For classification models only.  The value corresponding to model output when the target event occurs.

    Returns
    -------
    dict
        Metrics calculated for each dataset and formatted as expected by SAS
        Model Manager.

    """

    datasets = (valid, train, test)

    labels = ['VALIDATE', 'TRAIN', 'TEST']

    results = []

    # At least some combination of datasets must be provided
    if all(d is None for d in datasets):
        raise ValueError("At least one dataset must be provided.")

    if event is None:
        event = 1
    elif event in model.classes_:
        event = model.classes_.index(event)
    else:
        event = int(event)

    for idx, dataset in enumerate(datasets):
        if dataset is None:
            continue

        X, y_true = dataset
        y_pred = model.predict(X)
        is_classification = hasattr(model, 'classes_')
        is_binary_classification = is_classification and len(model.classes_) == 2

        if is_classification:
            y_true = _convert_class_labels(model, y_true)
            y_pred = _convert_class_labels(model, y_pred)

            y_pred_probs = pd.DataFrame(model.predict_proba(X))
            y_prob = y_pred_probs.iloc[:, event]

        # Initialize all metrics to a default value
        ase = None
        rase = None
        ks = None
        auc = None
        gini = None
        mce = None
        mcll = None
        gamma = None
        tau = None

        h = logging.FileHandler('test.log')
        log.setLevel(logging.DEBUG)
        log.addHandler(h)
        try:
            if is_binary_classification:
                ase = mean_squared_error(y_true, y_prob)
            elif is_classification:
                # Do not calculate ASE/RMSE for multiclass problems until SAS MM numbers can be matched
                pass
            else:
                ase = mean_squared_error(y_true, y_pred)
            rase = sqrt(ase)
        except (ValueError, TypeError):
            log.debug('Unable to calculate RMSE:', exc_info=1)

        # Kolmogorov - Smirnov (KS) Statistics
        try:
            if is_classification:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                ks = max(np.abs(fpr - tpr))
        except ValueError:
            log.debug('Unable to calculate KS statistic:', exc_info=1)

        # Area under Curve
        try:
            if is_classification and len(model.classes_) == 2:
                auc = roc_auc_score(y_true, y_prob)
                gini = 2 * auc - 1
        except (ValueError, TypeError):
            log.debug('Unable to calculate AUC and Gini index:', exc_info=1)

        # Misclassification Rate
        try:
            if is_binary_classification:
                mce = 1 - accuracy_score(y_true, y_pred)
        except ValueError:
            log.debug('Unable to calculate Misclassificationrate:', exc_info=1)

        # Multi-Class Log Loss
        try:
            if is_binary_classification:
                mcll = log_loss(y_true, y_pred_probs)
        except ValueError:
            log.debug('Unable to calculate Multi-class Log Loss:', exc_info=1)

        # Gamma
        # try:
        #     if is_classification:
        #         from scipy.stats import gamma
        #         _, _, beta = gamma.fit(y_pred)
        #         gamma = 1. / beta
        # except ImportError:
        #     log.debug('Unable to calculate Gamma distribution:', exc_info=1)

        # Tau
        # try:
        #     if is_classification:
        #         from scipy.stats import kendalltau
        #         tau, _ = kendalltau(y_true, y_pred)
        # except Exception:
        #     log.debug('Unable to calculate Kendall Tau coefficient:', exc_info=1)

        stats = {
            '_DataRole_': labels[idx],
            '_PartInd_': str(idx),
            '_PartInd__f': '           %d' % idx,
            '_NObs_': len(y_pred),
            '_DIV_': len(y_pred),
            '_ASE_': ase,
            '_C_': auc,
            '_RASE_': rase,
            '_GAMMA_': gamma,
            '_GINI_': gini,
            '_KSPostCutoff_': None,
            '_KS_': ks,
            '_KSCut_': None,
            '_MCE_': mce,
            '_MCLL_': mcll,
            '_TAU_': tau
        }

        results.append({'dataMap': stats, 'rowNumber': idx, 'header': None})

    return {'data': results}
