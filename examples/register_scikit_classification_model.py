#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from sasctl import Session, register_model, publish_model
from sklearn.linear_model import LogisticRegression


# Load the Iris data set and split into features and target.
df = pd.read_csv('data/iris.csv')
X = df.drop('Species', axis=1)
y = df.Species.astype('category')

# Fit a sci-kit learn model
model = LogisticRegression(max_iter=10000)
model.fit(X, y)

# Establish a session with Viya
with Session('hostname', 'username', 'password'):
    model_name = 'Iris Regression'

    # Register the model in Model Manager
    register_model(model,
                   model_name,
                   input=X,         # Use X to determine model inputs
                   project='Iris',  # Register in "Iris" project
                   force=True)      # Create project if it doesn't exist

    # Publish the model to the real-time scoring engine
    module = publish_model(model_name, 'maslocal')

    # Select the first row of training data
    x = X.iloc[0, :]

    # Call the published module and score the record
    result = module.predict(x)
    print(result)