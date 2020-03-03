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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, ClassifierMixin


def prep_rocstat(model, X_train, y_train, X_test, y_test, targetname, outdir, templatedir):
    '''
    Function to prepare ROC json file
    '''

    # print("Printing...")

    # Define CustomThreshold class
    class CustomThreshold(BaseEstimator, ClassifierMixin):
        """ Custom threshold wrapper for binary classification"""

        def __init__(self, base, threshold=0.5):
            self.base = base
            self.threshold = threshold

        def fit(self, *args, **kwargs):
            self.base.fit(*args, **kwargs)
            return self

        def predict(self, X):
            return (self.base.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    # Prepare sample
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.3, random_state=12345)

    # Load template
    with open(templatedir + "/dmcas_roc_template_class.json", "r") as jsonFile:
        body = json.load(jsonFile)

    # Create measures lists
    th = range(0, 105, 5)
    th_final = []
    count_th = 0

    for i in th:
        th_final.append(round(i / 100, 2))

    count_row = 0
    part_list = (2, 1, 0)

    for j in part_list:

        clf = [CustomThreshold(model, threshold) for threshold in list(th_final)]

        row = 0

        for _ in range(0, 110, 5):

            stats = dict()

            # define datasets
            if j == 1:  # train
                X_part = X_t
                y_part = y_t
                stats.update(_DataRole_='TRAIN')
                stats.update(_PartInd_="1")
                stats.update(_PartInd__f="           1")

            elif j == 0:  # validation
                X_part = X_v
                y_part = y_v
                stats.update(_DataRole_='VALIDATE')
                stats.update(_PartInd_="0")
                stats.update(_PartInd__f="           0")

            elif j == 2:  # test
                X_part = X_test
                y_part = y_test
                stats.update(_DataRole_='TEST')
                stats.update(_PartInd_="2")
                stats.update(_PartInd__f="           2")

            if (row + 1) == 22:
                stats.update(_ACC_=None)
                stats.update(_TP_=None)
                stats.update(_OneMinusSpecificity_="0")
                stats.update(_Column_='P_' + str(targetname) + '1')
                stats.update(_TN_=None)
                stats.update(_KS2_=None)
                stats.update(_FPR_="0")
                stats.update(_FDR_="0")
                stats.update(_MiscEvent_=None)
                stats.update(_FN_=None)
                stats.update(_KS_=None)
                stats.update(_Sensitivity_="0")
                stats.update(_Event_="1")
                stats.update(_FP_=None)
                stats.update(_Cutoff_="1")
                stats.update(_Specificity_="1")
                stats.update(_FHALF_=None)

            else:
                # score model for each threshold
                y_score = clf[row].predict(X_part)

                stats.update(_Cutoff_=str(th_final[row]))

                # confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_part, y_score).ravel()
                stats.update(_TP_=str(tp))
                stats.update(_TN_=str(tn))
                stats.update(_FN_=str(fn))
                stats.update(_FP_=str(fp))

                # sensitivity, hit rate, recall, or true positive rate
                tpr = tp / (tp + fn)
                stats.update(_Sensitivity_=str(tpr))

                # specificity or true negative rate
                tnr = tn / (tn + fp)

                # precision or positive predictive value
                # ppv = tp/(tp+fp)

                # negative predictive value
                # npv = tn/(tn+fn)

                # fall out or false positive rate
                fpr = fp / (fp + tn)
                stats.update(_OneMinusSpecificity_=str(fpr))
                stats.update(_FPR_=str(fpr))
                stats.update(_Specificity_=str(fpr + 1))

                # false negative rate
                fnr = fn / (tp + fn)

                # false discovery rate
                fdr = fp / (tp + fp)
                stats.update(_FDR_=str(fdr))

                # overall accuracy
                acc = (tp + tn) / (tp + fp + fn + tn)
                stats.update(_ACC_=str(acc))
                stats.update(_MiscEvent_=str(1 - acc))

                # F0.5 score
                f_05 = fbeta_score(y_part, y_score, beta=0.5)
                stats.update(_FHALF_=str(f_05))

                # KS statistics on 2 samples
                y_score_arr = np.array(y_score)
                y_score_df = pd.DataFrame(data=y_score_arr)
                y_test_arr = np.array(y_part)
                y_test_df = pd.DataFrame(data=y_test_arr)
                KS_2 = ks_2samp(y_test_df[0], y_score_df[0])
                stats.update(_KS2_=str(KS_2[0]))

                # misc
                stats.update(_Column_='P_' + str(targetname) + '1')
                stats.update(_KS_=str(max(tpr - fpr, 0)))  # to check
                stats.update(_Event_="1")

            body['data'][count_row] = {'dataMap': stats}
            body['data'][count_row]['rowNumber'] = str(count_row + 1)
            body['data'][count_row]['header'] = None

            row += 1
            count_row += 1

    with open(outdir + '/dmcas_roc.json', 'w') as f:
        json.dump(body, f, indent=2)

    print("Saved in:", outdir)


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

    # prepare function

    # def calc_cumulative_gains(df: pd.DataFrame, actual_col: str, predicted_col: str, probability_col: str):
    #
    #     df.sort_values(by=probability_col, ascending=False, inplace=True)
    #
    #     subset = df[df[predicted_col] == True]
    #
    #     rows = []
    #
    #     for group in np.array_split(subset, 20):
    #         score = accuracy_score(group[actual_col].tolist(),
    #                                        group[predicted_col].tolist(),
    #                                        normalize=False)
    #
    #         rows.append({'NumCases': len(group),
    #                      'NumCorrectPredictions': score})
    #
    #     lift = pd.DataFrame(rows)
    #
    #     # Cumulative Gains Calculation
    #
    #     lift['RunningCorrect'] = lift['NumCorrectPredictions'].cumsum()
    #     lift['PercentCorrect'] = lift.apply(
    #         lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x['RunningCorrect'], axis=1)
    #     lift['CumulativeCorrectBestCase'] = lift['NumCases'].cumsum()
    #     lift['PercentCorrectBestCase'] = lift['CumulativeCorrectBestCase'].apply(
    #         lambda x: 100 if (100 / lift['NumCorrectPredictions'].sum()) * x > 100 else (100 / lift[
    #             'NumCorrectPredictions'].sum()) * x)
    #     lift['AvgCase'] = lift['NumCorrectPredictions'].sum() / len(lift)
    #     lift['CumulativeAvgCase'] = lift['AvgCase'].cumsum()
    #     lift['PercentAvgCase'] = lift['CumulativeAvgCase'].apply(
    #         lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x)
    #
    #     # Lift Chart
    #     lift['NormalisedPercentAvg'] = 1
    #     lift['NormalisedPercentWithModel'] = lift['PercentCorrect'] / \
    #                                          lift['PercentAvgCase']
    #
    #     lift_dict = lift.to_dict()
    #
    #     return lift_dict

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

        # TODO: bin rows by percentiles or evenly?
        num_groups = 20
        groups = []
        for i, g in enumerate(np.array_split(df, num_groups)):
            stats = {'nobjs': len(g),
                     'true_events': (g['target'] == event).count(),
                     'accuracy': accuracy_score(g['target'],
                                                g['predicted'], normalize=False),
                     'sample': i}
            groups.append(stats)

        t2 = pd.DataFrame(groups)
        t2['total_correct'] = t2['accuracy'].cumsum()
        t2['total_obs'] = t2['nobjs'].cumsum()
        t2['percent_accuracy'] = 100 * t2['total_correct'] / t2['total_obs']

        total_true_events = (df['target'] == event).sum()
        total_samples = len(df)
        true_event_rate = total_true_events / float(total_samples)
        # -----

        actualValue = df['target']
        predictValue = df['P_Type1']
        numObservations = len(actualValue)
        quantileCutOff = np.percentile(df['P_Type1'], np.arange(0, 100, 10))


    return {'data': None}


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
