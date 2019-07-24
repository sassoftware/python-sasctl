#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import swat

from sasctl import Session
from sasctl.tasks import register_model, publish_model
from sasctl.services import microanalytic_score as mas

s = swat.CAS('hostname', 'username', 'password')
s.loadactionset('decisionTree')

tbl = s.CASTable('iris')
tbl.decisiontree.gbtreetrain(target='Species',
                             inputs=['SepalLength', 'SepalWidth',
                                     'PetalLength', 'PetalWidth'],
                             savestate='gradboost_astore')

astore = s.CASTable('gradboost_astore')

with Session(s):
    model = register_model(astore, 'Gradient Boosting', 'Iris')
    module = publish_model(model, 'maslocal')
    response = mas.execute_module_step(module, 'score',
                                       SepalLength=5.1,
                                       SepalWidth=3.5,
                                       PetalLength=1.4,
                                       PetalWidth=0.2)
