#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_fit_statistics_with_hmeq():
    pd = pytest.importorskip('pandas')

    df = pd.DataFrame(dict(y_true=[1,
                                   1,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0,
                                   1,
                                   1,
                                   0,
                                   0,
                                   1,
                                   0,
                                   0,
                                   1,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0,
                                   1,
                                   1,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   1,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0],
                           y_pred=[0.117132672, 0.108227341, 0.035062903
                               , 0.110345952
                               , 0.028506742
                               , 0.029971723
                               , 0.028548193
                               , 0.48679085
                               , 0.103400456
                               , 0.036782007
                               , 0.07732951
                               , 0.81185122
                               , 0.02742061
                               , 0.005141424
                               , 0.041517529
                               , 0.004962792
                               , 0.024052988
                               , 0.029369196
                               , 0.028402176
                               , 0.005858391
                               , 0.074853227
                               , 0.070015485
                               , 0.242709738
                               , 0.004901838
                               , 0.158428103
                               , 0.005025242
                               , 0.153849503
                               , 0.141071872
                               , 0.183443247
                               , 0.132565627
                               , 0.567097207
                               , 0.252146736
                               , 0.130812749
                               , 0.169175214
                               , 0.023379282
                               , 0.12332713
                               , 0.075391336
                               , 0.132924488
                               , 0.078480089
                               , 0.134787114
                               , 0.666707409
                               , 0.216909301
                               , 0.118089377
                               , 0.015758684
                               , 0.125863115
                               , 0.67515834
                               , 0.052953516
                               , 0.137602522
                               , 0.019715027
                               , 0.126108901
                               , 0.018378935
                               , 0.016198776
                               , 0.013881948
                               , 0.110090943
                               , 0.19655504
                               , 0.086394764
                               , 0.117675834]))

    from sasctl.utils.metrics import compare
    stats = compare.fit_statistics(df.y_true, df.y_pred)

def test_fit_statistics_with_iris(iris_dataset):
    sklearn = pytest.importorskip('sklearn')

    from sklearn.ensemble import RandomForestClassifier
    from sasctl.utils.metrics import compare

    model = RandomForestClassifier()
    X = iris_dataset.drop('Species', axis=1)
    y_true = iris_dataset['Species']
    model.fit(X, y_true)
    y_hat = model.predict(X)

    stats = compare.fit_statistics(y_true, y_hat)

    assert isinstance(stats, dict)

    # Should only contain stats for training data
    assert len(stats['data']) == 1

    assert stats['data'][0]['rowNumber'] == 1
    assert stats['data'][0]['dataMap']['_DataRole_'] == 'TRAIN'
    assert stats['data'][0]['dataMap']['_NObs_'] == X.shape[0]
    assert stats['data'][0]['dataMap']['_DIV_'] == X.shape[0]

    for stat in ('_MCE_', ):
        assert stats['data'][0]['dataMap'][stat] is not None

    # Should have:
    # RASE
    # MCE
    # ASE
    # MCLL
    # MiscEvent ???
    # C




