import math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import h2o

h2o.init()

model = h2o.import_mojo(str(Path("/models/resources/viya/4c5dd027-d442-4860-9e96-9c26060dc727/glmfit_mojo.mojo")))

def score(LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):
    "Output: EM_CLASSIFICATION, EM_EVENTPROBABILITY"

    try:
        global model
    except NameError:
        model = h2o.import_mojo(str(Path("/models/resources/viya/4c5dd027-d442-4860-9e96-9c26060dc727/glmfit_mojo.mojo")))

    try:
        if math.isnan(LOAN):
            LOAN = 18724.518290980173
    except TypeError:
        LOAN = 18724.518290980173
    try:
        if math.isnan(MORTDUE):
            MORTDUE = 73578.70182374542
    except TypeError:
        MORTDUE = 73578.70182374542
    try:
        if math.isnan(VALUE):
            VALUE = 102073.94160831199
    except TypeError:
        VALUE = 102073.94160831199
    try:
        REASON = REASON.strip()
    except AttributeError:
        REASON = ""
    try:
        JOB = JOB.strip()
    except AttributeError:
        JOB = ""
    try:
        if math.isnan(YOJ):
            YOJ = 8.878919914084074
    except TypeError:
        YOJ = 8.878919914084074
    try:
        if math.isnan(DEROG):
            DEROG = 0.2522264631043257
    except TypeError:
        DEROG = 0.2522264631043257
    try:
        if math.isnan(DELINQ):
            DELINQ = 0.4452373565001551
    except TypeError:
        DELINQ = 0.4452373565001551
    try:
        if math.isnan(CLAGE):
            CLAGE = 179.86044681046295
    except TypeError:
        CLAGE = 179.86044681046295
    try:
        if math.isnan(NINQ):
            NINQ = 1.1648318042813455
    except TypeError:
        NINQ = 1.1648318042813455
    try:
        if math.isnan(CLNO):
            CLNO = 21.205105889178995
    except TypeError:
        CLNO = 21.205105889178995
    try:
        if math.isnan(DEBTINC):
            DEBTINC = 33.64816965600249
    except TypeError:
        DEBTINC = 33.64816965600249

    input_array = pd.DataFrame([[LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],
                               columns=["LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"],
                               dtype=object,
                               index=[0])
    column_types = {"LOAN" : "numeric", "MORTDUE" : "numeric", "VALUE" : "numeric", "REASON" : "string", "JOB" : "string", "YOJ" : "numeric", "DEROG" : "numeric", "DELINQ" : "numeric", "CLAGE" : "numeric", "NINQ" : "numeric", "CLNO" : "numeric", "DEBTINC" : "numeric"}
    h2o_array = h2o.H2OFrame(input_array, column_types=column_types)
    prediction = model.predict(h2o_array)
    prediction = h2o.as_list(prediction, use_pandas=False)
    EM_CLASSIFICATION = prediction[1][0]
    EM_EVENTPROBABILITY = float(prediction[1][1])

    return EM_CLASSIFICATION, EM_EVENTPROBABILITY