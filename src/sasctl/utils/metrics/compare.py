#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import json

from random import randrange, uniform, gauss
from math import sqrt
from scipy.stats import kendalltau
from sklearn.base import TransformerMixin
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, ClassifierMixin



def fit_statistics(train_expected=None, train_actual=None,
                   valid_expected=None, valid_actual=None,
                   test_expected=None, test_actual=None):



    for expected, actual in ((train_expected, train_actual), ):
        if expected is None or actual is None:
            continue

        metrics = {}

        metrics['_DataRole_'] = '' # TRAIN / VALIDATE / TEST

        # Number of observations used to calculate metrics
        metrics['_NObs_'] = len(actual)

        # Root Average Squared Error
        metrics['_RASE_'] = sqrt(metrics.mean_squared_error(expected, actual))


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

    def calc_cumulative_gains(df: pd.DataFrame, actual_col: str, predicted_col: str, probability_col: str):

        df.sort_values(by=probability_col, ascending=False, inplace=True)

        subset = df[df[predicted_col] == True]

        rows = []

        for group in np.array_split(subset, 20):
            score = metrics.accuracy_score(group[actual_col].tolist(),
                                           group[predicted_col].tolist(),
                                           normalize=False)

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

        lift_dict = lift.to_dict()

        return lift_dict

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
            score = metrics.accuracy_score(
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


# def fit_statistics(model, X_train, y_train, X_test, y_test, outdir, templatedir):
def fit_statistics(train_expected=None, train_actual=None,
                       valid_expected=None, valid_actual=None,
                       test_expected=None, test_actual=None):

    datasets = ((train_expected, train_actual), (valid_expected, valid_actual),
                (test_expected, test_actual))

    labels = ['TEST', 'VALIDATE', 'TRAIN']

    t0 = all(d is None for d in datasets)

    for expected, actual in datasets:
        label = labels.pop()

        if expected is None or actual is None:
            continue

        # Average Squared Error
        ase = metrics.mean_squared_error(expected, actual)

        # Root Average Squared Error
        rase = sqrt(ase)

        fpr, tpr, _ = metrics.roc_curve(expected, actual)

        # Area under Curve
        auc = metrics.roc_auc_score(expected, actual)
        gini = (2 * auc) - 1

        #
        mce = 1 - metrics.accuracy_score(expected, actual)

        # Multi-Class Log Loss
        mcll = metrics.log_loss((expected, actual))

        # KS uses the Kolmogorov-Smirnov coefficient as the objective function
        stats.update(_KS_=str(max(fpr - tpr)))


        stats = {
            '_DataRole_': label,
            '_NObs_': len(actual),
            '_ASE_': ase,
            '_C_': auc,
            '_RASE_': rase,
            '_GINI_': gini,
            '_KS_': None,
            '_MCE_': mce,
            '_MCLL_': mcll
        }

        # GAMMA uses the gamma coefficient as the objective function
        # from scipy.stats import gamma
        # shape, loc, scale = gamma.fit(model.predict(updata), floc=0)
        # _GAMMA_ = 1/scale

        # GAMMA (Method of Moments) uses the gamma coefficient as the objective function

        def calculateGammaParams(data):
            mean = np.mean(data)
            std = np.std(data)
            shape = (mean / std) ** 2
            scale = (std ** 2) / mean
            return (shape, 0, scale)
    #
    #     eshape, eloc, escale = calculateGammaParams(data[i][1])
    #     stats.update(_GAMMA_=str(1 / escale))
    #
    #
    #     stats.update(_formattedPartition_='           ' + str(i))
    #
    #
    #
    #     # _KSPostCutoff_
    #
    #     stats.update(_KSPostCutoff_='null')
    #
    #     # _DIV_
    #
    #     stats.update(_DIV_=len(data[i][0]))
    #
    #     # TAU uses the tau coefficient as the objective function
    #
    #     stats.update(_TAU_=str(kendalltau(data[i][0], data[i][1])[0]))
    #
    #     # C uses Area Under ROC
    #
    #     stats.update(_C_=str(metrics.auc(fpr, tpr)))
    #
    #     # _KSCut_
    #
    #     stats.update(_KSCut_='null')
    #
    #     _PartInd_ = str(i)
    #
    #     # rowNumber
    #
    #     stats.update(rowNumber=str(i))
    #
    #     # header
    #
    #     stats.update(header='null')
    #
    #     body['data'][i] = {'dataMap': stats}
    #
    # with open(outdir + '/dmcas_fitstat.json', 'w') as f:
    #     json.dump(body, f, indent=2)
    #
    # print("Saved in:", outdir)