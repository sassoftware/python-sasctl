import math
import pickle
import pandas as pd
import numpy as np
import settings

def scoreDecisionTreeClassifier(Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    try:
        _thisModelFit
    except NameError:
        with open(settings.pickle_path + 'DecisionTreeClassifier.pickle', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)

    try:
        if math.isnan(Speed_sensor):
            Speed_sensor = 75.06777416636571
    except TypeError:
        Speed_sensor = 75.06777416636571

    try:
        if math.isnan(Vibration):
            Vibration = 246.95994479089921
    except TypeError:
        Vibration = 246.95994479089921

    try:
        if math.isnan(Engine_Load):
            Engine_Load = 36.06875427988444
    except TypeError:
        Engine_Load = 36.06875427988444

    try:
        if math.isnan(Coolant_Temp):
            Coolant_Temp = 87.7360057782593
    except TypeError:
        Coolant_Temp = 87.7360057782593

    try:
        if math.isnan(Intake_Pressure):
            Intake_Pressure = 119.41133983387505
    except TypeError:
        Intake_Pressure = 119.41133983387505

    try:
        if math.isnan(Engine_RPM):
            Engine_RPM = 1263.593956903816
    except TypeError:
        Engine_RPM = 1263.593956903816

    try:
        if math.isnan(Speed_OBD):
            Speed_OBD = 75.06777416636571
    except TypeError:
        Speed_OBD = 75.06777416636571

    try:
        if math.isnan(Intake_Air):
            Intake_Air = 10.664138678223186
    except TypeError:
        Intake_Air = 10.664138678223186

    try:
        if math.isnan(Flow_Rate):
            Flow_Rate = 23.769685807150594
    except TypeError:
        Flow_Rate = 23.769685807150594

    try:
        if math.isnan(Throttle_Pos):
            Throttle_Pos = 75.25847583170821
    except TypeError:
        Throttle_Pos = 75.25847583170821

    try:
        if math.isnan(Voltage):
            Voltage = 14.171672083784758
    except TypeError:
        Voltage = 14.171672083784758

    try:
        if math.isnan(Ambient):
            Ambient = 7.309738774527507
    except TypeError:
        Ambient = 7.309738774527507

    try:
        if math.isnan(Accel):
            Accel = 22.004593668231607
    except TypeError:
        Accel = 22.004593668231607

    try:
        if math.isnan(Engine_Oil_Temp):
            Engine_Oil_Temp = 84.52690501986277
    except TypeError:
        Engine_Oil_Temp = 84.52690501986277

    try:
        if math.isnan(Speed_GPS):
            Speed_GPS = 77.28586848607259
    except TypeError:
        Speed_GPS = 77.28586848607259

    try:
        if math.isnan(GPS_Longitude):
            GPS_Longitude = 8.569920139091247
    except TypeError:
        GPS_Longitude = 8.569920139091247

    try:
        if math.isnan(GPS_Latitude):
            GPS_Latitude = 48.54705087261828
    except TypeError:
        GPS_Latitude = 48.54705087261828

    try:
        if math.isnan(GPS_Bearing):
            GPS_Bearing = 195.5093294811605
    except TypeError:
        GPS_Bearing = 195.5093294811605

    try:
        if math.isnan(GPS_Altitude):
            GPS_Altitude = 326.22607439508846
    except TypeError:
        GPS_Altitude = 326.22607439508846

    try:
        if math.isnan(Turbo_Boost):
            Turbo_Boost = 3.2505074667148186
    except TypeError:
        Turbo_Boost = 3.2505074667148186

    try:
        if math.isnan(Trip_Distance):
            Trip_Distance = 230.49551875887803
    except TypeError:
        Trip_Distance = 230.49551875887803

    try:
        if math.isnan(Litres_Per_km):
            Litres_Per_km = 4.316090813719474
    except TypeError:
        Litres_Per_km = 4.316090813719474

    try:
        if math.isnan(Accel_Ssor_Total):
            Accel_Ssor_Total = -0.0031587095220898037
    except TypeError:
        Accel_Ssor_Total = -0.0031587095220898037

    try:
        if math.isnan(CO2):
            CO2 = 93.42142250399041
    except TypeError:
        CO2 = 93.42142250399041

    try:
        if math.isnan(Trip_Time):
            Trip_Time = 7512.937041049717
    except TypeError:
        Trip_Time = 7512.937041049717

    try:
        if math.isnan(CO_emission):
            CO_emission = 0.0
    except TypeError:
        CO_emission = 0.0

    try:
        if math.isnan(HC_emission):
            HC_emission = 0.0
    except TypeError:
        HC_emission = 0.0

    try:
        if math.isnan(PM_emission):
            PM_emission = 0.0
    except TypeError:
        PM_emission = 0.0

    try:
        if math.isnan(NOx_emission):
            NOx_emission = 0.0
    except TypeError:
        NOx_emission = 0.0

    try:
        if math.isnan(CO2_emission):
            CO2_emission = 0.0
    except TypeError:
        CO2_emission = 0.0

    try:
        if math.isnan(Fuel_level):
            Fuel_level = 0.0
    except TypeError:
        Fuel_level = 0.0

    try:
        if math.isnan(Oil_life):
            Oil_life = 0.0
    except TypeError:
        Oil_life = 0.0

    try:
        if math.isnan(Vibration_alert):
            Vibration_alert = 1.0
    except TypeError:
        Vibration_alert = 1.0

    try:
        if math.isnan(VibrationAlert_Total):
            VibrationAlert_Total = 123.0
    except TypeError:
        VibrationAlert_Total = 123.0

    try:
        if math.isnan(Vibration_Recent):
            Vibration_Recent = 12.0
    except TypeError:
        Vibration_Recent = 12.0

    try:
        if math.isnan(Turbo_alert):
            Turbo_alert = 1.0
    except TypeError:
        Turbo_alert = 1.0

    try:
        if math.isnan(Emission_alert):
            Emission_alert = 1.0
    except TypeError:
        Emission_alert = 1.0

    try:
        if math.isnan(Fog_control):
            Fog_control = 1.0
    except TypeError:
        Fog_control = 1.0

    try:
        if math.isnan(Engine_control):
            Engine_control = 1.0
    except TypeError:
        Engine_control = 1.0

    inputArray = pd.DataFrame([[1.0, Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control]],
                              columns = ['const', 'Speed_sensor', 'Vibration', 'Engine_Load', 'Coolant_Temp', 'Intake_Pressure', 'Engine_RPM', 'Speed_OBD', 'Intake_Air', 'Flow_Rate', 'Throttle_Pos', 'Voltage', 'Ambient', 'Accel', 'Engine_Oil_Temp', 'Speed_GPS', 'GPS_Longitude', 'GPS_Latitude', 'GPS_Bearing', 'GPS_Altitude', 'Turbo_Boost', 'Trip_Distance', 'Litres_Per_km', 'Accel_Ssor_Total', 'CO2', 'Trip_Time', 'CO_emission', 'HC_emission', 'PM_emission', 'NOx_emission', 'CO2_emission', 'Fuel_level', 'Oil_life', 'Vibration_alert', 'VibrationAlert_Total', 'Vibration_Recent', 'Turbo_alert', 'Emission_alert', 'Fog_control', 'Engine_control'],
                              dtype = float)

    prediction = _thisModelFit.predict(inputArray)

    EM_EVENTPROBABILITY = float(prediction)

    if (EM_EVENTPROBABILITY >= 0.2269170579029734):
        EM_CLASSIFICATION = '1'
    else:
        EM_CLASSIFICATION = '0' 

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)