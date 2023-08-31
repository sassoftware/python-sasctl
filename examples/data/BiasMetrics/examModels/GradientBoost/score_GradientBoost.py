import math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "LinearRegression.pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

import math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "LinearRegression.pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

def score(gender_male, raceethnicity_group_A, raceethnicity_group_B, raceethnicity_group_C, raceethnicity_group_D, parental_level_of_education_associates_degree, parental_level_of_education_bachelors_degree, parental_level_of_education_high_school, parental_level_of_education_some_college, parental_level_of_education_some_high_school, lunch_freereduced, test_preparation_course_none):
    "Output: EM_PREDICTION"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "LinearRegression.pickle", "rb") as pickle_model:
                model = pickle.load(pickle_model)



    try:
        gender_male = gender_male.strip()
    except AttributeError:
        gender_male = ""
    try:
        raceethnicity_group_A = raceethnicity_group_A.strip()
    except AttributeError:
        raceethnicity_group_A = ""
    try:
        raceethnicity_group_B = raceethnicity_group_B.strip()
    except AttributeError:
        raceethnicity_group_B = ""
    try:
        raceethnicity_group_C = raceethnicity_group_C.strip()
    except AttributeError:
        raceethnicity_group_C = ""
    try:
        raceethnicity_group_D = raceethnicity_group_D.strip()
    except AttributeError:
        raceethnicity_group_D = ""
    try:
        parental_level_of_education_associates_degree = parental_level_of_education_associates_degree.strip()
    except AttributeError:
        parental_level_of_education_associates_degree = ""
    try:
        parental_level_of_education_bachelors_degree = parental_level_of_education_bachelors_degree.strip()
    except AttributeError:
        parental_level_of_education_bachelors_degree = ""
    try:
        parental_level_of_education_high_school = parental_level_of_education_high_school.strip()
    except AttributeError:
        parental_level_of_education_high_school = ""
    try:
        parental_level_of_education_some_college = parental_level_of_education_some_college.strip()
    except AttributeError:
        parental_level_of_education_some_college = ""
    try:
        parental_level_of_education_some_high_school = parental_level_of_education_some_high_school.strip()
    except AttributeError:
        parental_level_of_education_some_high_school = ""
    try:
        lunch_freereduced = lunch_freereduced.strip()
    except AttributeError:
        lunch_freereduced = ""
    try:
        test_preparation_course_none = test_preparation_course_none.strip()
    except AttributeError:
        test_preparation_course_none = ""

    input_array = pd.DataFrame([[gender_male, raceethnicity_group_A, raceethnicity_group_B, raceethnicity_group_C, raceethnicity_group_D, parental_level_of_education_associates_degree, parental_level_of_education_bachelors_degree, parental_level_of_education_high_school, parental_level_of_education_some_college, parental_level_of_education_some_high_school, lunch_freereduced, test_preparation_course_none]],
                              columns=["gender_male", "raceethnicity_group_A", "raceethnicity_group_B", "raceethnicity_group_C", "raceethnicity_group_D", "parental_level_of_education_associates_degree", "parental_level_of_education_bachelors_degree", "parental_level_of_education_high_school", "parental_level_of_education_some_college", "parental_level_of_education_some_high_school", "lunch_freereduced", "test_preparation_course_none"],
                              dtype=float)
    prediction = model.predict(input_array)

    # Check for numpy values and convert to a CAS readable representation
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()[0]

    EM_PREDICTION = prediction

    return EM_PREDICTIONimport math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "RandomForest.pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

def score(gender_male, raceethnicity_group_A, raceethnicity_group_B, raceethnicity_group_C, raceethnicity_group_D, parental_level_of_education_associates_degree, parental_level_of_education_bachelors_degree, parental_level_of_education_high_school, parental_level_of_education_some_college, parental_level_of_education_some_high_school, lunch_freereduced, test_preparation_course_none):
    "Output: EM_PREDICTION"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "RandomForest.pickle", "rb") as pickle_model:
                model = pickle.load(pickle_model)



    try:
        gender_male = gender_male.strip()
    except AttributeError:
        gender_male = ""
    try:
        raceethnicity_group_A = raceethnicity_group_A.strip()
    except AttributeError:
        raceethnicity_group_A = ""
    try:
        raceethnicity_group_B = raceethnicity_group_B.strip()
    except AttributeError:
        raceethnicity_group_B = ""
    try:
        raceethnicity_group_C = raceethnicity_group_C.strip()
    except AttributeError:
        raceethnicity_group_C = ""
    try:
        raceethnicity_group_D = raceethnicity_group_D.strip()
    except AttributeError:
        raceethnicity_group_D = ""
    try:
        parental_level_of_education_associates_degree = parental_level_of_education_associates_degree.strip()
    except AttributeError:
        parental_level_of_education_associates_degree = ""
    try:
        parental_level_of_education_bachelors_degree = parental_level_of_education_bachelors_degree.strip()
    except AttributeError:
        parental_level_of_education_bachelors_degree = ""
    try:
        parental_level_of_education_high_school = parental_level_of_education_high_school.strip()
    except AttributeError:
        parental_level_of_education_high_school = ""
    try:
        parental_level_of_education_some_college = parental_level_of_education_some_college.strip()
    except AttributeError:
        parental_level_of_education_some_college = ""
    try:
        parental_level_of_education_some_high_school = parental_level_of_education_some_high_school.strip()
    except AttributeError:
        parental_level_of_education_some_high_school = ""
    try:
        lunch_freereduced = lunch_freereduced.strip()
    except AttributeError:
        lunch_freereduced = ""
    try:
        test_preparation_course_none = test_preparation_course_none.strip()
    except AttributeError:
        test_preparation_course_none = ""

    input_array = pd.DataFrame([[gender_male, raceethnicity_group_A, raceethnicity_group_B, raceethnicity_group_C, raceethnicity_group_D, parental_level_of_education_associates_degree, parental_level_of_education_bachelors_degree, parental_level_of_education_high_school, parental_level_of_education_some_college, parental_level_of_education_some_high_school, lunch_freereduced, test_preparation_course_none]],
                              columns=["gender_male", "raceethnicity_group_A", "raceethnicity_group_B", "raceethnicity_group_C", "raceethnicity_group_D", "parental_level_of_education_associates_degree", "parental_level_of_education_bachelors_degree", "parental_level_of_education_high_school", "parental_level_of_education_some_college", "parental_level_of_education_some_high_school", "lunch_freereduced", "test_preparation_course_none"],
                              dtype=float)
    prediction = model.predict(input_array)

    # Check for numpy values and convert to a CAS readable representation
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()[0]

    EM_PREDICTION = prediction

    return EM_PREDICTIONimport math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "GradientBoost.pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

def score(gender_male, raceethnicity_group_A, raceethnicity_group_B, raceethnicity_group_C, raceethnicity_group_D, parental_level_of_education_associates_degree, parental_level_of_education_bachelors_degree, parental_level_of_education_high_school, parental_level_of_education_some_college, parental_level_of_education_some_high_school, lunch_freereduced, test_preparation_course_none):
    "Output: EM_PREDICTION"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "GradientBoost.pickle", "rb") as pickle_model:
                model = pickle.load(pickle_model)



    try:
        gender_male = gender_male.strip()
    except AttributeError:
        gender_male = ""
    try:
        raceethnicity_group_A = raceethnicity_group_A.strip()
    except AttributeError:
        raceethnicity_group_A = ""
    try:
        raceethnicity_group_B = raceethnicity_group_B.strip()
    except AttributeError:
        raceethnicity_group_B = ""
    try:
        raceethnicity_group_C = raceethnicity_group_C.strip()
    except AttributeError:
        raceethnicity_group_C = ""
    try:
        raceethnicity_group_D = raceethnicity_group_D.strip()
    except AttributeError:
        raceethnicity_group_D = ""
    try:
        parental_level_of_education_associates_degree = parental_level_of_education_associates_degree.strip()
    except AttributeError:
        parental_level_of_education_associates_degree = ""
    try:
        parental_level_of_education_bachelors_degree = parental_level_of_education_bachelors_degree.strip()
    except AttributeError:
        parental_level_of_education_bachelors_degree = ""
    try:
        parental_level_of_education_high_school = parental_level_of_education_high_school.strip()
    except AttributeError:
        parental_level_of_education_high_school = ""
    try:
        parental_level_of_education_some_college = parental_level_of_education_some_college.strip()
    except AttributeError:
        parental_level_of_education_some_college = ""
    try:
        parental_level_of_education_some_high_school = parental_level_of_education_some_high_school.strip()
    except AttributeError:
        parental_level_of_education_some_high_school = ""
    try:
        lunch_freereduced = lunch_freereduced.strip()
    except AttributeError:
        lunch_freereduced = ""
    try:
        test_preparation_course_none = test_preparation_course_none.strip()
    except AttributeError:
        test_preparation_course_none = ""

    input_array = pd.DataFrame([[gender_male, raceethnicity_group_A, raceethnicity_group_B, raceethnicity_group_C, raceethnicity_group_D, parental_level_of_education_associates_degree, parental_level_of_education_bachelors_degree, parental_level_of_education_high_school, parental_level_of_education_some_college, parental_level_of_education_some_high_school, lunch_freereduced, test_preparation_course_none]],
                              columns=["gender_male", "raceethnicity_group_A", "raceethnicity_group_B", "raceethnicity_group_C", "raceethnicity_group_D", "parental_level_of_education_associates_degree", "parental_level_of_education_bachelors_degree", "parental_level_of_education_high_school", "parental_level_of_education_some_college", "parental_level_of_education_some_high_school", "lunch_freereduced", "test_preparation_course_none"],
                              dtype=float)
    prediction = model.predict(input_array)

    # Check for numpy values and convert to a CAS readable representation
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()[0]

    EM_PREDICTION = prediction

    return EM_PREDICTION