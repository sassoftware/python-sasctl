

import math
import pickle
import pandas as pd
import numpy as np


global _thisModelFit

with open('/models/resources/viya/256cb2d0-b91a-4a39-a5c2-3548a527a26a/RandomForest.pickle', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pfile)

def scoreRandomForest(LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    try:
        _thisModelFit
    except NameError:

        with open('/models/resources/viya/256cb2d0-b91a-4a39-a5c2-3548a527a26a/RandomForest.pickle', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)

    try:
        inputArray = pd.DataFrame([[LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],
                                  columns=['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'],
                                  dtype=float)
        prediction = _thisModelFit.predict(inputArray)
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required
    # For example, many statsmodels models include an intercept value that must be included for the model prediction
        inputArray = pd.DataFrame([[1.0, LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],
                                columns=['const', 'LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'],
                                dtype=float)
        prediction = _thisModelFit.predict(inputArray)

    try:
        EM_EVENTPROBABILITY = float(prediction)
    except TypeError:
    # If the model expects non-binary responses, a TypeError will be raised.
    # The except block shifts the prediction to accept a non-binary response.
        EM_EVENTPROBABILITY = float(prediction[:,1])

    if (EM_EVENTPROBABILITY >= 0.199496644295302):
        EM_CLASSIFICATION = '1'
    else:
        EM_CLASSIFICATION = '0' 

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)