#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import sklearn.datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sasctl import Session
from sasctl.tasks import register_model, publish_model
from sasctl.services import model_repository as mr
from sasctl.services import model_management as mm


data = sklearn.datasets.load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Establish a session with SAS Viya
Session('hostname', 'username', 'password')

project = 'Boston Housing'
model_name = 'Boston Regression'

# Fit a linear regression model using sci-kit learn
lm = LinearRegression()
lm.fit(X_train, y_train)

# Register the model in SAS Model Manager
register_model(lm,
               model_name,
               input=X_train,       # Use X to determine model inputs
               project=project,     # Register in "Iris" project
               force=True)          # Create project if it doesn't exist

# Update project properties
project = mr.get_project(project)
project['function'] = 'prediction'
project['targetLevel'] = 'interval'
project['targetVariable'] = 'Price'
project['predictionVariable'] = 'var1'
project = mr.update_project(project)

# Instruct the project to look for tables in the "Public" CAS library with
# names starting with "boston_" and use these tables to track model
# performance over time.
mm.create_performance_definition(model_name, 'Public', 'boston')

# Publish the model to the real-time scoring engine
module_lm = publish_model(model_name, 'maslocal')

# Select the first row of training data
x = X.iloc[0, :]

# Call the published module and score the record
result = module_lm.score(**x)
print(result)

# Build a second model
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# Register the second model in Model Manager
model_dt = register_model(dt, 'Decision Tree', project, input=X)

# Publish from Model Manager -> MAS
module_dt = publish_model(model_dt, 'maslocal')

# Use MAS to score some new data
result = module_dt.score(**x)
print(result)



