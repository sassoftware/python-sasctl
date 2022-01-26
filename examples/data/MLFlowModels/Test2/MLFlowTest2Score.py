

import math
import pickle
import pandas as pd
import numpy as np

import settings

with open(settings.pickle_path + 'MLFlowTest2.pickle', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)

def scoreMLFlowTest2(fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol):
    "Output: tensor"

    try:
        global _thisModelFit
    except NameError:

        with open(settings.pickle_path + 'MLFlowTest2.pickle', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)

    try:
        inputArray = pd.DataFrame([[fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol]],
                                  columns=['fixedacidity', 'volatileacidity', 'citricacid', 'residualsugar', 'chlorides', 'freesulfurdioxide', 'totalsulfurdioxide', 'density', 'pH', 'sulphates', 'alcohol'],
                                  dtype=float)
        prediction = _thisModelFit.predict(inputArray)
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required
    # For example, many statsmodels models include an intercept value that must be included for the model prediction
        inputArray = pd.DataFrame([[1.0, fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol]],
                                columns=['const', 'fixedacidity', 'volatileacidity', 'citricacid', 'residualsugar', 'chlorides', 'freesulfurdioxide', 'totalsulfurdioxide', 'density', 'pH', 'sulphates', 'alcohol'],
                                dtype=float)
        prediction = _thisModelFit.predict(inputArray)

    tensor = prediction
    if isinstance(tensor, np.ndarray):
        tensor = prediction.item(0)

    return(tensor)
