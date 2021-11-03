

import math
import pickle
import pandas as pd
import numpy as np

with open('/models/resources/viya/cb485d04-6cb9-48ce-a829-67b8167303ce/LinearRegression.pickle', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)

def scoreLinearRegression(Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population):
    "Output: EM_PREDICTION, EM_PREDICTION"

    try:
        global _thisModelFit
    except NameError:

        with open('/models/resources/viya/cb485d04-6cb9-48ce-a829-67b8167303ce/LinearRegression.pickle', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)

    try:
        inputArray = pd.DataFrame([[Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population]],
                                  columns=['Avg_Area_Income', 'Avg_Area_House_Age', 'Avg_Area_Number_of_Rooms', 'Avg_Area_Number_of_Bedrooms', 'Area_Population'],
                                  dtype=float)
        prediction = _thisModelFit.predict(inputArray)
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required
    # For example, many statsmodels models include an intercept value that must be included for the model prediction
        inputArray = pd.DataFrame([[1.0, Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population]],
                                columns=['const', 'Avg_Area_Income', 'Avg_Area_House_Age', 'Avg_Area_Number_of_Rooms', 'Avg_Area_Number_of_Bedrooms', 'Area_Population'],
                                dtype=float)
        prediction = _thisModelFit.predict(inputArray)

    try:
        EM_PREDICTION = float(prediction)
    except TypeError:
    # If the model expects non-binary responses, a TypeError will be raised.
    # The except block shifts the prediction to accept a non-binary response.
        EM_PREDICTION = float(prediction[:,1])

    if (EM_PREDICTION >= 1232072.6541453):
        EM_PREDICTION = '1'
    else:
        EM_PREDICTION = '0' 

    return(EM_PREDICTION, EM_PREDICTION)
