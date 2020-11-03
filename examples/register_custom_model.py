#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from datetime import datetime

from sasctl import register_model, Session


s = Session('hostname', 'username', 'password')

# The register_model task will attempt to extract the necessary metadata from the provided ASTORE file or Python model.
# However, if this doesn't work for your model or you need to specify different metadata, you can provide it as a
# dictionary instead.  For a full list of parameters that can be specified see the documentation here:
# https://developer.sas.com/apis/rest/DecisionManagement/#schemamodel
model_info = {
    'name': 'Custom Model',
    'description': 'This model is for demonstration purposes only.',
    'scoreCodeType': 'Python',
    'algorithm': 'Other'
}

# To include the contents of the model itself, simply provide the information for each file in a list.
files = [

    # Files can be added to the model by specifying a name of the file and its contents
    dict(name='greetings.txt', file='Hello World!'),

    # You can also specify file-like object to be included.  Here we upload this Python file itself to the model.
    # In addition, the optional `role` parameter can be used to assign a File Role to the file in Model Manager.
    dict(name=__file__, file=open(__file__), role='Score code'),

    # The files also need not be simple text.  Here we create a simple Python datetime object, pickle it, and then
    # include the binary file with the model.
    dict(name='datetime.pkl', file=pickle.dumps(datetime.now()))
]

model = register_model(model_info, name=model_info['name'], project='Examples', files=files, force=True)
