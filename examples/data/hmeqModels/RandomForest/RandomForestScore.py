import math
import pickle
import pandas as pd
import numpy as np

import settings
from pathlib import Path

with open(Path(settings.pickle_path) / 'RandomForest.pickle', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)

def scoreRandomForest(LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    try:
        global _thisModelFit
    except NameError:

        with open(Path(settings.pickle_path) / 'RandomForest.pickle', 'rb') as _pFile:
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
        # If the prediction returns as a list of values or improper value type, a TypeError will be raised.
        # Attempt to handle the prediction output in the except block.
        EM_EVENTPROBABILITY = float(prediction[0])

    if (EM_EVENTPROBABILITY >= 0.199496644295302):
        EM_CLASSIFICATION = '1'
    else:
        EM_CLASSIFICATION = '0' 

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)
