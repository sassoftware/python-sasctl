#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from sasctl import Session, register_model, publish_model
from sklearn.ensemble import GradientBoostingRegressor


# Convert the local CSV file into a Pandas DataFrame
df = pd.read_csv('data/boston_house_prices.csv')

# The model input data (X) is every column in the DataFrame except the target.
# The target (y) is equal to the median home value.
target = 'medv'
X = df.drop(target, axis=1)
y = df[target]

# Fit a sci-kit learn model
model = GradientBoostingRegressor()
model.fit(X, y)

# Establish a session with Viya
with Session('hostname', 'username', 'password'):
    model_name = 'GB Regression'
    project_name = 'Boston Housing'

    # Register the model in SAS Model Manager
    register_model(model, model_name, project_name, input=X, force=True)

    # Publish the model to the real-time scoring engine
    module = publish_model(model_name, 'maslocal', replace=True)

    # Select the first row of training data
    x = X.iloc[0, :]

    # Call the published module and score the record
    result = module.predict(x)
    print(result)