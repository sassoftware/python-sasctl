import h2o
import gzip, shutil, os

import math
import pickle
import pandas as pd
import numpy as np


global _thisModelFit

h2o.init()

_thisModelFit = h2o.load_model('/models/resources/viya/cd6cd3c5-174f-4f51-9f43-92a1d695b0e7/glmFit.pickle')

def scoreglmFit(LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    try:
        _thisModelFit
    except NameError:

        _thisModelFit = h2o.load_model('/models/resources/viya/cd6cd3c5-174f-4f51-9f43-92a1d695b0e7/glmFit.pickle')

    inputArray = pd.DataFrame([[LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],
                              columns=['LOAN', 'MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'],
                              dtype=float, index=[0])
    columnTypes = {'LOAN':'numeric', 'MORTDUE':'numeric', 'VALUE':'numeric', 'REASON':'numeric', 'JOB':'numeric', 'YOJ':'numeric', 'DEROG':'numeric', 'DELINQ':'numeric', 'CLAGE':'numeric', 'NINQ':'numeric', 'CLNO':'numeric', 'DEBTINC':'numeric'}
    h2oArray = h2o.H2OFrame(inputArray, column_types=columnTypes)
    prediction = _thisModelFit.predict(h2oArray)
    prediction = h2o.as_list(prediction, use_pandas=False)

    EM_EVENTPROBABILITY = float(prediction[1][2])
    EM_CLASSIFICATION = prediction[1][0]

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)