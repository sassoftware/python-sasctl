#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
import json

from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, log_loss, roc_auc_score
from sklearn.metrics import accuracy_score, mean_squared_error


def prep_liftstat(model, X_train, y_train, X_test, y_test, targetname, targetvalue, outdir, templatedir):
    '''
    Function to prepare Liftstat json file
    https://go.documentation.sas.com/?cdcId=vdmmlcdc&cdcVersion=1.0&docsetId=casml&docsetTarget=viyaml_assess_examples01.htm&locale=en
    '''

    # prepare sample

    X_t, X_v, y_t, y_v = train_test_split(
        X_train, y_train, test_size=0.3, random_state=12345)
    data = [(X_t, y_t, model.predict_proba(X_t)),
            (X_v, y_v, model.predict_proba(X_v)),
            (X_test, y_test, model.predict_proba(X_test))]

    with open(templatedir + '/R_HMEQ_lift.json', 'r') as i:

        body = json.load(i)

    count = 0

    for j in range(3):

        # prepare data

        data[j][0].reset_index(drop=True, inplace=True)
        data[j][1].reset_index(drop=True, inplace=True)
        score_probability = pd.DataFrame(data[j][2], columns=[
            'P_' + str(targetname) + '0', 'P_' + str(targetname) + '1'])
        df = pd.concat([data[j][0], data[j][1], score_probability], axis=1)
        df['pred_col'] = np.where(
            df['P_' + str(targetname) + '1'] >= 0.5, 1, 0)

        df.sort_values(by='P_' + str(targetname) + '1',
                       ascending=False, inplace=True)

        subset = df[df['pred_col'] == True]

        rows = []

        for group in np.array_split(subset, 20):
            score = accuracy_score(
                group[targetname].tolist(), group['pred_col'].tolist(), normalize=False)

            rows.append({'NumCases': len(group),
                         'NumCorrectPredictions': score})

            lift = pd.DataFrame(rows)

            # Cumulative Gains Calculation

            lift['RunningCorrect'] = lift['NumCorrectPredictions'].cumsum()
            lift['PercentCorrect'] = lift.apply(
                lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x['RunningCorrect'], axis=1)
            lift['CumulativeCorrectBestCase'] = lift['NumCases'].cumsum()
            lift['PercentCorrectBestCase'] = lift['CumulativeCorrectBestCase'].apply(
                lambda x: 100 if (100 / lift['NumCorrectPredictions'].sum()) * x > 100 else (100 / lift[
                    'NumCorrectPredictions'].sum()) * x)
            lift['AvgCase'] = lift['NumCorrectPredictions'].sum() / len(lift)
            lift['CumulativeAvgCase'] = lift['AvgCase'].cumsum()
            lift['PercentAvgCase'] = lift['CumulativeAvgCase'].apply(
                lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x)

            # Lift Chart
            lift['NormalisedPercentAvg'] = 1
            lift['NormalisedPercentWithModel'] = lift['PercentCorrect'] / \
                                                 lift['PercentAvgCase']
            lift['Gain'] = lift['NormalisedPercentWithModel'] - lift['NormalisedPercentAvg']

            lift_dict = lift.to_dict()

        #             return lift_dict

        #         calc_cumulative_gains(perf_probability, targetname, 'pred_col', 'P_' + str(targetname) + '1')

        for index, row in enumerate(range(0, 105, 5)):

            count = count + 1

            stats = dict()

            if j == 0:
                stats.update(_DataRole_='VALIDATE')
            if j == 1:
                stats.update(_DataRole_='TRAIN')
            if j == 2:
                stats.update(_DataRole_='TEST')

            if index == 0:

                stats.update(_CumLiftBest_=None)
                stats.update(_NObs_='0')
                stats.update(_Depth_='0')
                stats.update(_CumRespBest_='0')
                stats.update(_formattedPartition_='           0')
                stats.update(_Column_=str(targetname))
                stats.update(_Lift_=None)
                stats.update(_RespBest_='0')
                stats.update(_CumPctResp_=None)
                stats.update(_NEventsBest_='0')
                stats.update(_CumResp_='0')
                stats.update(_PctRespBest_=None)
                stats.update(_Gain_=None)
                stats.update(_CumLift_='0')
                stats.update(_CumPctRespBest_=None)
                stats.update(_Value_='0')
                stats.update(_Event_=str(targetvalue))
                stats.update(_NEvents_='0')
                stats.update(_PctResp_=None)
                stats.update(_LiftBest_=None)
                stats.update(_PartInd_='0')
                stats.update(_Resp_='0')
                stats.update(_GainBest_=None)

            else:
                stats.update(_CumLiftBest_=str(lift_dict['NormalisedPercentWithModel'][index - 1]))
                stats.update(_NObs_=str(lift_dict['NumCases'][index - 1]))
                stats.update(_Depth_=str(index - 1))
                stats.update(_CumRespBest_='100')
                stats.update(_formattedPartition_='           0')
                stats.update(_Column_=str(targetname))
                stats.update(_Lift_=str(lift_dict['NormalisedPercentWithModel'][index - 1]))
                stats.update(_RespBest_=str(lift_dict['PercentCorrect'][index - 1]))
                stats.update(_CumPctResp_=str(lift_dict['PercentCorrectBestCase'][index - 1]))
                stats.update(_NEventsBest_=str(lift_dict['NumCorrectPredictions'][index - 1]))
                stats.update(_CumResp_=str(lift_dict['RunningCorrect'][index - 1]))
                stats.update(_PctRespBest_=str(lift_dict['PercentCorrect'][index - 1]))
                stats.update(_Gain_=str(lift['Gain'][index - 1]))
                stats.update(_CumLift_=str(lift_dict['RunningCorrect'][index - 1]))
                stats.update(_CumPctRespBest_=str(lift_dict['PercentCorrect'][index - 1]))
                stats.update(_Value_=str(lift['Gain'][index - 1]))
                stats.update(_Event_=str(targetvalue))
                stats.update(_NEvents_=str(' '))
                stats.update(_PctResp_=str(lift_dict['PercentCorrect'][index - 1]))
                stats.update(_LiftBest_=str(lift_dict['NormalisedPercentWithModel'][index - 1]))
                stats.update(_PartInd_=str(j))
                stats.update(_Resp_=str(lift_dict['PercentCorrect'][index - 1]))
                stats.update(_GainBest_=str(lift['Gain'][index - 1]))

            body['data'][count - 1] = {'dataMap': stats}
            body['data'][count - 1]['rowNumber'] = count
            body['data'][count - 1]['header'] = None

    with open(outdir + '/dmcas_lift.json', 'w') as f:
        json.dump(body, f, indent=2)

    print("Saved in:", outdir)


def lift_statistics(model, train=None, valid=None, test=None, event=None):
    """

    Parameters
    ----------
    model
    train
    valid
    test
    event

    Returns
    -------

    """
    datasets = (valid, train, test)
    labels = ['VALIDATE', 'TRAIN', 'TEST']

    # At least some combination of datasets must be provided
    if all(d is None for d in datasets):
        raise ValueError("At least one dataset must be provided.")

    results = []
    row_count = 0
    if event is None:
        event = 1
    elif event in model.classes_:
        event = 0 if event == model.classes_[0] else 1
    else:
        event = int(event)

    for idx, dataset in enumerate(datasets):
        if dataset is None:
            continue

        X, y_true = dataset

        target_column = getattr(y_true, 'name', 'Class')
        proba_columns = ['P_%s%d' % (target_column, i) for i in (0, 1)]
        event_column = proba_columns[event]

        # We need to re-assign int values to the dataset to ensure they match
        # up to the column order output by the model.  Otherwise, output labels
        # will match, but underlying int representations will be off.
        y_true = y_true.cat.reorder_categories(model.classes_)

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

        # Total # of observations in each group
        obs_counts_by_group = label_counts_by_group.sum(axis=1)

        # Percent of total observations in each group
        obs_percent_by_group = 100 * obs_counts_by_group / total_samples

        # Number of events in full dataset
        total_event_count = event_counts_by_group.sum()

        # Percent of total events included in group
        gain_by_group = 100 * event_counts_by_group / total_event_count

        # Percent of obs in group that are event
        event_percent_by_group = 100 * event_counts_by_group / obs_counts_by_group

        overallResponsePercent = 100 * (total_event_count / total_samples)
        lift = event_percent_by_group / overallResponsePercent

        liftDf = pd.DataFrame({'Quantile Number': obs_counts_by_group,  # 10/20/etc
                               'Quantile Percent': obs_percent_by_group, #
                               'Gain Number': event_counts_by_group,
                               'Gain Percent': gain_by_group,
                               'Response Percent': event_percent_by_group,
                               'Lift': lift})

        accCountTable = label_counts_by_group.cumsum(axis = 0)
        obs_counts_by_group = accCountTable.sum(1)
        obs_percent_by_group = 100 * (obs_counts_by_group / total_samples)
        event_counts_by_group = accCountTable[event]
        gain_by_group = 100 * (event_counts_by_group / total_event_count)
        event_percent_by_group = 100 * (event_counts_by_group / obs_counts_by_group)
        accLift = event_percent_by_group / overallResponsePercent

        accLiftDf = pd.DataFrame({'Acc Quantile Number': obs_counts_by_group,
                                  'Acc Quantile Percent': obs_percent_by_group,
                                  'Acc Gain Number': event_counts_by_group,
                                  'Acc Gain Percent': gain_by_group,
                                  'Acc Response Percent': event_percent_by_group,
                                  'Acc Lift': accLift})

        liftDf = pd.concat([liftDf, accLiftDf], axis=1, ignore_index=False)

        for index, row in liftDf.iterrows():
            row_count += 1

            stats = {'header': None,
                     'rowNumber': row_count,
                     'dataMap': {'_DataRole_': labels[idx],
                                 '_Column_': target_column,
                                 '_Event_': event,
                                 '_Depth_': row['Acc Quantile Percent'],
                                 '_NObs_': row['Quantile Number'],
                                 '_Gain_': row['Gain Number'],      # of events in bucket
                                 '_Resp_': row['Gain Percent'],       # Capture Response
                                 '_CumResp_': row['Acc Gain Percent'],
                                 '_PctResp_': row['Response Percent'],       # % of bin that is event
                                 '_CumPctResp_': row['Acc Response Percent'],    # % of events in all bins so far
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
    model
    train
    valid
    test

    Returns
    -------

    """
    datasets = (valid, train, test)
    labels = ['VALIDATE', 'TRAIN', 'TEST']

    # At least some combination of datasets must be provided
    if all(d is None for d in datasets):
        raise ValueError("At least one dataset must be provided.")

    results = []
    row_count = 0

    for idx, dataset in enumerate(datasets):
        if dataset is None:
            continue

        X, y_true = dataset

        y_pred = model.predict(X)
        y_pred = pd.Series(y_pred, dtype=y_true.dtype)

        fpr, tpr, threshold = roc_curve(y_true.cat.codes, y_pred.cat.codes)

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
                         '_Cutoff_': h,
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


def fit_statistics(model, train=None, valid=None, test=None):
    """Calculate model fit statistics.

    Parameters
    ----------
    model
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

    results = []

    # At least some combination of datasets must be provided
    if all(d is None for d in datasets):
        raise ValueError("At least one dataset must be provided.")

    for idx, dataset in enumerate(datasets):
        if dataset is None:
            continue

        X, y_true = dataset
        y_pred = model.predict(X)

        # Average Squared Error
        try:
            ase = mean_squared_error(y_true, y_pred)
            rase = sqrt(ase)
        except ValueError:
            ase = None
            rase = None

        try:
            # Kolmogorov - Smirnov (KS) Statistics
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            ks = max(np.abs(fpr - tpr))
        except ValueError:
            ks = None

        # Area under Curve
        try:
            auc = roc_auc_score(y_true, y_pred)
            gini = (2 * auc) - 1
        except (ValueError, TypeError):
            auc = None
            gini = None

        try:
            # Misclassification Error
            mce = 1 - accuracy_score(y_true, y_pred)  # classification
        except ValueError:
            mce = None

        # Multi-Class Log Loss
        try:
            mcll = log_loss(y_true, y_pred)
        except ValueError:
            mcll = None

        # KS uses the Kolmogorov-Smirnov coefficient as the objective function
        # stats.update(_KS_=str(max(fpr - tpr)))

        stats = {
            '_DataRole_': labels[idx],
            '_PartInd_': str(idx),
            '_PartInd__f': '           %d' % idx,
            '_NObs_': len(y_pred),
            '_DIV_': len(y_pred),
            '_ASE_': ase,
            '_C_': auc,
            '_RASE_': rase,
            '_GINI_': gini,
            '_KSPostCutoff_': None,
            '_KS_': ks,
            '_KSCut_': None,
            '_MCE_': mce,
            '_MCLL_': mcll
        }

        results.append({'dataMap': stats, 'rowNumber': idx, 'header': None})

    return {'data': results}
