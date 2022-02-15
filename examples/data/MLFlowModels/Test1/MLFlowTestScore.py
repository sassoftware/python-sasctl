

import math
import pickle
import pandas as pd
import numpy as np

with open('/models/resources/viya/d02aadfe-618e-44e0-af6d-1bf00c6396e3/MLFlowTest.pickle', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)

def scoreMLFlowTest(fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol):
    "Output: tensor"

    try:
        global _thisModelFit
    except NameError:

        with open('/models/resources/viya/d02aadfe-618e-44e0-af6d-1bf00c6396e3/MLFlowTest.pickle', 'rb') as _pFile:
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

    return(tensor)
