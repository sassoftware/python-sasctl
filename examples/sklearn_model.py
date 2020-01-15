#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from sasctl import Session, register_model, publish_model


# Load the Iris data set and convert into a Pandas data frame.
raw = datasets.load_iris()
X = pd.DataFrame(raw.data, columns=['SepalLength', 'SepalWidth',
                                    'PetalLength', 'PetalWidth'])
y = pd.DataFrame(raw.target, columns=['Species'], dtype='category')
y.Species.cat.categories = raw.target_names

# Fit a sci-kit learn model
model = LogisticRegression()
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
    result = module.predict(**x)
    print(result)