#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import swat
from sasctl import Session
from sasctl.tasks import register_model, publish_model


# Connect to the CAS server
s = swat.CAS('hostname', 5570, 'username', 'password')

# Upload the training data to CAS
tbl = s.upload('data/iris.csv').casTable

# Train a gradient boosting model to identify iris species.
s.loadactionset('decisionTree')
tbl.decisionTree.gbtreetrain(target='Species',
                             inputs=['SepalLength', 'SepalWidth',
                                     'PetalLength', 'PetalWidth'],
                             savestate='gradboost_astore')

# Establish a reference to the newly created ASTORE table.
astore = s.CASTable('gradboost_astore')

# Connect to the SAS environment
with Session('hostname', 'username', 'password'):
    # Register the trained model by providing:
    #  - the ASTORE containing the model
    #  - a name for the model
    #  - a name for the project
    #
    # NOTE: the force=True option will create the project if it does not exist.
    model = register_model(astore, 'Gradient Boosting', 'Iris', force=True)

    # Publish the model to SAS® Micro Analytic Service (MAS).  Specifically to
    # the default MAS service "maslocal".
    module = publish_model(model, 'maslocal')

    # sasctl wraps the published module with Python methods corresponding to
    # the various steps defined in the module (like "predict").
    response = module.score(SepalLength=5.1, SepalWidth=3.5,
                            PetalLength=1.4, PetalWidth=0.2)
