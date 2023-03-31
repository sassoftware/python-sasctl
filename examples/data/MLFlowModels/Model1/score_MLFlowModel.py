import math
import cloudpickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "MLFlowTest.pickle", "rb") as pickle_model:
    model = cloudpickle.load(pickle_model)

def score(fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol):
    "Output: tensor"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "MLFlowTest.pickle", "rb") as pickle_model:
                model = cloudpickle.load(pickle_model)

    input_array = pd.DataFrame([[fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol]],
                              columns=["fixedacidity", "volatileacidity", "citricacid", "residualsugar", "chlorides", "freesulfurdioxide", "totalsulfurdioxide", "density", "pH", "sulphates", "alcohol"],
                              dtype=float)
    prediction = model.predict(input_array)

    # Check for numpy values and convert to a CAS readable representation
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()[0]

    tensor = prediction

    return tensor