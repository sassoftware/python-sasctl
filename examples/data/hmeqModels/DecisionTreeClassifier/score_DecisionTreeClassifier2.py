import math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "DecisionTreeClassifier2.pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

def score(LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):
    "Output: EM_CLASSIFICATION, EM_EVENTPROBABILITY"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "DecisionTreeClassifier2.pickle", "rb") as pickle_model:
            model = pickle.load(pickle_model)


    index=None
    if not isinstance(LOAN, pd.Series):
        index=[0]
    input_array = pd.DataFrame(
        {"LOAN": LOAN, "MORTDUE": MORTDUE, "VALUE": VALUE, "YOJ": YOJ, "DEROG": DEROG,
        "DELINQ": DELINQ, "CLAGE": CLAGE, "NINQ": NINQ, "CLNO": CLNO, "DEBTINC":
        DEBTINC}, index=index
    )
    input_array = impute_missing_values(input_array)
    prediction = model.predict_proba(input_array).tolist()

    # Check for numpy values and convert to a CAS readable representation
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()

    if input_array.shape[0] == 1:
        if prediction[0][1] > 0.5:
            EM_CLASSIFICATION = "1"
        else:
            EM_CLASSIFICATION = "0"
        return EM_CLASSIFICATION, prediction[0][1]
    else:
        df = pd.DataFrame(prediction)
        proba = df[1]
        classifications = np.where(df[1] > 0.5, '1', '0')
        return pd.DataFrame({'EM_CLASSIFICATION': classifications, 'EM_EVENTPROBABILITY': proba})

def impute_missing_values(data):
    impute_values = \
        {'VALUE': 101776.04874145007, 'DEBTINC': 33.779915349235246, 'NINQ':
        1.1860550458715597, 'DELINQ': 0.4494423791821561, 'CLAGE': 179.7662751900465,
        'MORTDUE': 73760.817199559, 'DEROG': 0.2545696877380046, 'YOJ':
        8.922268135904499, 'CLNO': 21.29609620076682, 'LOAN': 18607.96979865772}
    return data.replace('           .', np.nan).fillna(impute_values).apply(pd.to_numeric, errors='ignore')
