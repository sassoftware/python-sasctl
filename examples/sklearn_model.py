#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from sasctl import Session, register_model


raw = datasets.load_iris()
X = pd.DataFrame(raw.data, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
y = pd.DataFrame(raw.target, columns=['Species'], dtype='category')
y.Species.cat.categories = raw.target_names

model = LogisticRegression()
model.fit(X, y)

with Session('example.com', user='username', password='*****'):
    register_model(model, 'Logistic Regression', project='Iris', force=True)
