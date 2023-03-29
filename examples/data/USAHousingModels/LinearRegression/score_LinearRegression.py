import math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "LinearRegression.pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

def score(Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population):
    "Output: EM_PREDICTION"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "LinearRegression.pickle", "rb") as pickle_model:
                model = pickle.load(pickle_model)

    try:
        if math.isnan(Avg_Area_Income):
            Avg_Area_Income = 68583.10898397
    except TypeError:
        Avg_Area_Income = 68583.10898397
    try:
        if math.isnan(Avg_Area_House_Age):
            Avg_Area_House_Age = 5.977222035287
    except TypeError:
        Avg_Area_House_Age = 5.977222035287
    try:
        if math.isnan(Avg_Area_Number_of_Rooms):
            Avg_Area_Number_of_Rooms = 6.9877918509092005
    except TypeError:
        Avg_Area_Number_of_Rooms = 6.9877918509092005
    try:
        if math.isnan(Avg_Area_Number_of_Bedrooms):
            Avg_Area_Number_of_Bedrooms = 3.9813300000000003
    except TypeError:
        Avg_Area_Number_of_Bedrooms = 3.9813300000000003
    try:
        if math.isnan(Area_Population):
            Area_Population = 36163.516038540256
    except TypeError:
        Area_Population = 36163.516038540256

    input_array = pd.DataFrame([[Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population]],
                              columns=["Avg_Area_Income", "Avg_Area_House_Age", "Avg_Area_Number_of_Rooms", "Avg_Area_Number_of_Bedrooms", "Area_Population"],
                              dtype=float)
    prediction = model.predict(input_array)

    # Check for numpy values and convert to a CAS readable representation
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()[0]

    EM_PREDICTION = prediction

    return EM_PREDICTION