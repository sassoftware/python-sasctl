import math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "GradientBoosting.pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

def score(LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):
    "Output: EM_CLASSIFICATION, EM_EVENTPROBABILITY"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "GradientBoosting.pickle", "rb") as pickle_model:
                model = pickle.load(pickle_model)



    try:
        if math.isnan(LOAN):
            LOAN = 18607.96979865772
    except TypeError:
        LOAN = 18607.96979865772
    try:
        if math.isnan(MORTDUE):
            MORTDUE = 73760.817199559
    except TypeError:
        MORTDUE = 73760.817199559
    try:
        if math.isnan(VALUE):
            VALUE = 101776.04874145007
    except TypeError:
        VALUE = 101776.04874145007
    try:
        if math.isnan(YOJ):
            YOJ = 8.922268135904499
    except TypeError:
        YOJ = 8.922268135904499
    try:
        if math.isnan(DEROG):
            DEROG = 0.2545696877380046
    except TypeError:
        DEROG = 0.2545696877380046
    try:
        if math.isnan(DELINQ):
            DELINQ = 0.4494423791821561
    except TypeError:
        DELINQ = 0.4494423791821561
    try:
        if math.isnan(CLAGE):
            CLAGE = 179.7662751900465
    except TypeError:
        CLAGE = 179.7662751900465
    try:
        if math.isnan(NINQ):
            NINQ = 1.1860550458715597
    except TypeError:
        NINQ = 1.1860550458715597
    try:
        if math.isnan(CLNO):
            CLNO = 21.29609620076682
    except TypeError:
        CLNO = 21.29609620076682
    try:
        if math.isnan(DEBTINC):
            DEBTINC = 33.779915349235246
    except TypeError:
        DEBTINC = 33.779915349235246

    input_array = pd.DataFrame([[LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],
                              columns=["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"],
                              dtype=float)
    prediction = model.predict_proba(input_array)

    # Check for numpy values and convert to a CAS readable representation
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()[0]

    if prediction[0] > prediction[1]:
        EM_CLASSIFICATION = "1"
    else:
        EM_CLASSIFICATION = "0"

    return EM_CLASSIFICATION, prediction[0]