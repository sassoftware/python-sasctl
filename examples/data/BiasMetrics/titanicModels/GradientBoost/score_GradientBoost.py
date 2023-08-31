import math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "GradientBoost.pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

def score(Age, SibSp, Parch, Fare, Sex_female, Sex_male, Pclass_1, Pclass_2, Pclass_3, Embarked_C, Embarked_Q, Embarked_S):
    "Output: EM_CLASSIFICATION, EM_EVENTPROBABILITY"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "GradientBoost.pickle", "rb") as pickle_model:
                model = pickle.load(pickle_model)



    try:
        if math.isnan(Age):
            Age = 28.06264
    except TypeError:
        Age = 28.06264
    try:
        if math.isnan(SibSp):
            SibSp = 0.712
    except TypeError:
        SibSp = 0.712
    try:
        if math.isnan(Parch):
            Parch = 0.464
    except TypeError:
        Parch = 0.464
    try:
        if math.isnan(Fare):
            Fare = 30.497600000000002
    except TypeError:
        Fare = 30.497600000000002
    try:
        Sex_female = Sex_female.strip()
    except AttributeError:
        Sex_female = ""
    try:
        Sex_male = Sex_male.strip()
    except AttributeError:
        Sex_male = ""
    try:
        Pclass_1 = Pclass_1.strip()
    except AttributeError:
        Pclass_1 = ""
    try:
        Pclass_2 = Pclass_2.strip()
    except AttributeError:
        Pclass_2 = ""
    try:
        Pclass_3 = Pclass_3.strip()
    except AttributeError:
        Pclass_3 = ""
    try:
        Embarked_C = Embarked_C.strip()
    except AttributeError:
        Embarked_C = ""
    try:
        Embarked_Q = Embarked_Q.strip()
    except AttributeError:
        Embarked_Q = ""
    try:
        Embarked_S = Embarked_S.strip()
    except AttributeError:
        Embarked_S = ""

    input_array = pd.DataFrame([[Age, SibSp, Parch, Fare, Sex_female, Sex_male, Pclass_1, Pclass_2, Pclass_3, Embarked_C, Embarked_Q, Embarked_S]],
                              columns=["Age", "SibSp", "Parch", "Fare", "Sex_female", "Sex_male", "Pclass_1", "Pclass_2", "Pclass_3", "Embarked_C", "Embarked_Q", "Embarked_S"],
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