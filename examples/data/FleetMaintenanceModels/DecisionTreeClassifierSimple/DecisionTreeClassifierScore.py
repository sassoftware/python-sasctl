

import math
import pickle
import pandas as pd
import numpy as np


global _thisModelFit

with open('/models/resources/viya/fc809ba9-11ec-45ad-bc49-ce2cd2a02fd4/DecisionTreeClassifier.pickle', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)

def scoreDecisionTreeClassifier(Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    try:
        _thisModelFit
    except NameError:

        with open('/models/resources/viya/fc809ba9-11ec-45ad-bc49-ce2cd2a02fd4/DecisionTreeClassifier.pickle', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)

    try:
        inputArray = pd.DataFrame([[Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control]],
                                  columns=['Speed_sensor', 'Vibration', 'Engine_Load', 'Coolant_Temp', 'Intake_Pressure', 'Engine_RPM', 'Speed_OBD', 'Intake_Air', 'Flow_Rate', 'Throttle_Pos', 'Voltage', 'Ambient', 'Accel', 'Engine_Oil_Temp', 'Speed_GPS', 'GPS_Longitude', 'GPS_Latitude', 'GPS_Bearing', 'GPS_Altitude', 'Turbo_Boost', 'Trip_Distance', 'Litres_Per_km', 'Accel_Ssor_Total', 'CO2', 'Trip_Time', 'CO_emission', 'HC_emission', 'PM_emission', 'NOx_emission', 'CO2_emission', 'Fuel_level', 'Oil_life', 'Vibration_alert', 'VibrationAlert_Total', 'Vibration_Recent', 'Turbo_alert', 'Emission_alert', 'Fog_control', 'Engine_control'],
                                  dtype=float)
        prediction = _thisModelFit.predict(inputArray)
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required
    # For example, many statsmodels models include an intercept value that must be included for the model prediction
        inputArray = pd.DataFrame([[1.0, Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control]],
                                columns=['const', 'Speed_sensor', 'Vibration', 'Engine_Load', 'Coolant_Temp', 'Intake_Pressure', 'Engine_RPM', 'Speed_OBD', 'Intake_Air', 'Flow_Rate', 'Throttle_Pos', 'Voltage', 'Ambient', 'Accel', 'Engine_Oil_Temp', 'Speed_GPS', 'GPS_Longitude', 'GPS_Latitude', 'GPS_Bearing', 'GPS_Altitude', 'Turbo_Boost', 'Trip_Distance', 'Litres_Per_km', 'Accel_Ssor_Total', 'CO2', 'Trip_Time', 'CO_emission', 'HC_emission', 'PM_emission', 'NOx_emission', 'CO2_emission', 'Fuel_level', 'Oil_life', 'Vibration_alert', 'VibrationAlert_Total', 'Vibration_Recent', 'Turbo_alert', 'Emission_alert', 'Fog_control', 'Engine_control'],
                                dtype=float)
        prediction = _thisModelFit.predict(inputArray)

    try:
        EM_EVENTPROBABILITY = float(prediction)
    except TypeError:
    # If the model expects non-binary responses, a TypeError will be raised.
    # The except block shifts the prediction to accept a non-binary response.
        EM_EVENTPROBABILITY = float(prediction[:,1])

    if (EM_EVENTPROBABILITY >= 0.2269170579029734):
        EM_CLASSIFICATION = '1'
    else:
        EM_CLASSIFICATION = '0' 

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)