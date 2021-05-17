import h2o
import gzip, shutil, os

import math
import pickle
import pandas as pd
import numpy as np

import settings


global _thisModelFit

h2o.init()

with gzip.open(settings.pickle_path + 'glmFit.mojo', 'r') as fileIn, open(settings.pickle_path + 'glmFit.' + '.zip', 'wb') as fileOut:
    shutil.copyfileobj(fileIn, fileOut)
os.chmod(settings.pickle_path + 'glmFit.' + '.zip', 0o777)
_thisModelFit = h2o.import_mojo(settings.pickle_path + 'glmFit.' + '.zip')

def scoreglmFit(Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    try:
        _thisModelFit
    except NameError:

_thisModelFit = h2o.import_mojo(settings.pickle_path + 'glmFit.' + '.zip')

    inputArray = pd.DataFrame([[Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control]],
                              columns=['Speed_sensor', 'Vibration', 'Engine_Load', 'Coolant_Temp', 'Intake_Pressure', 'Engine_RPM', 'Speed_OBD', 'Intake_Air', 'Flow_Rate', 'Throttle_Pos', 'Voltage', 'Ambient', 'Accel', 'Engine_Oil_Temp', 'Speed_GPS', 'GPS_Longitude', 'GPS_Latitude', 'GPS_Bearing', 'GPS_Altitude', 'Turbo_Boost', 'Trip_Distance', 'Litres_Per_km', 'Accel_Ssor_Total', 'CO2', 'Trip_Time', 'CO_emission', 'HC_emission', 'PM_emission', 'NOx_emission', 'CO2_emission', 'Fuel_level', 'Oil_life', 'Vibration_alert', 'VibrationAlert_Total', 'Vibration_Recent', 'Turbo_alert', 'Emission_alert', 'Fog_control', 'Engine_control'],
                              dtype=float, index=[0])
    columnTypes = {'Speed_sensor':'numeric', 'Vibration':'numeric', 'Engine_Load':'numeric', 'Coolant_Temp':'numeric', 'Intake_Pressure':'numeric', 'Engine_RPM':'numeric', 'Speed_OBD':'numeric', 'Intake_Air':'numeric', 'Flow_Rate':'numeric', 'Throttle_Pos':'numeric', 'Voltage':'numeric', 'Ambient':'numeric', 'Accel':'numeric', 'Engine_Oil_Temp':'numeric', 'Speed_GPS':'numeric', 'GPS_Longitude':'numeric', 'GPS_Latitude':'numeric', 'GPS_Bearing':'numeric', 'GPS_Altitude':'numeric', 'Turbo_Boost':'numeric', 'Trip_Distance':'numeric', 'Litres_Per_km':'numeric', 'Accel_Ssor_Total':'numeric', 'CO2':'numeric', 'Trip_Time':'numeric', 'CO_emission':'numeric', 'HC_emission':'numeric', 'PM_emission':'numeric', 'NOx_emission':'numeric', 'CO2_emission':'numeric', 'Fuel_level':'numeric', 'Oil_life':'numeric', 'Vibration_alert':'numeric', 'VibrationAlert_Total':'numeric', 'Vibration_Recent':'numeric', 'Turbo_alert':'numeric', 'Emission_alert':'numeric', 'Fog_control':'numeric', 'Engine_control':'numeric'}
    h2oArray = h2o.H2OFrame(inputArray, column_types=columnTypes)
    prediction = _thisModelFit.predict(h2oArray)
    prediction = h2o.as_list(prediction, use_pandas=False)

    EM_EVENTPROBABILITY = float(prediction[1][2])
    EM_CLASSIFICATION = prediction[1][0]

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)