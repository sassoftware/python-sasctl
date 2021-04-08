package pythonScore / overwrite=yes;
dcl package pymas pm;
dcl package logger logr('App.tk.MAS');
dcl varchar(32767) character set utf8 pypgm;
dcl int resultCode revision;

method score(double Speed_sensor, double Vibration, double Engine_Load, double Coolant_Temp, double Intake_Pressure, double Engine_RPM, double Speed_OBD, double Intake_Air, double Flow_Rate, double Throttle_Pos, double Voltage, double Ambient, double Accel, double Engine_Oil_Temp, double Speed_GPS, double GPS_Longitude, double GPS_Latitude, double GPS_Bearing, double GPS_Altitude, double Turbo_Boost, double Trip_Distance, double Litres_Per_km, double Accel_Ssor_Total, double CO2, double Trip_Time, double CO_emission, double HC_emission, double PM_emission, double NOx_emission, double CO2_emission, double Fuel_level, double Oil_life, double Vibration_alert, double VibrationAlert_Total, double Vibration_Recent, double Turbo_alert, double Emission_alert, double Fog_control, double Engine_control, in_out double resultCode, in_out double EM_EVENTPROBABILITY, in_out varchar(100) EM_CLASSIFICATION);
   resultCode = revision = 0;
   if null(pm) then do;
      pm = _new_ pymas();
      resultCode = pm.useModule('model_exec_02ed248b-1ff4-42ab-848c-57aa2dce04b8', 1);
      if resultCode then do;
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('import math');
         resultCode = pm.appendSrcLine('import pickle');
         resultCode = pm.appendSrcLine('import pandas as pd');
         resultCode = pm.appendSrcLine('import numpy as np');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('global _thisModelFit');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('with open("/models/resources/viya/fc809ba9-11ec-45ad-bc49-ce2cd2a02fd4/DecisionTreeClassifier.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('    _thisModelFit = pickle.load(_pFile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('def scoreDecisionTreeClassifier(Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control):');
         resultCode = pm.appendSrcLine('    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        _thisModelFit');
         resultCode = pm.appendSrcLine('    except NameError:');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('        with open("/models/resources/viya/fc809ba9-11ec-45ad-bc49-ce2cd2a02fd4/DecisionTreeClassifier.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('            _thisModelFit = pickle.load(_pFile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control]],');
         resultCode = pm.appendSrcLine('                                  columns=["Speed_sensor", "Vibration", "Engine_Load", "Coolant_Temp", "Intake_Pressure", "Engine_RPM", "Speed_OBD", "Intake_Air", "Flow_Rate", "Throttle_Pos", "Voltage", "Ambient", "Accel", "Engine_Oil_Temp", "Speed_GPS", "GPS_Longitude", "GPS_Latitude", "GPS_Bearing", "GPS_Altitude", "Turbo_Boost", "Trip_Distance", "Litres_Per_km", "Accel_Ssor_Total", "CO2", "Trip_Time", "CO_emission", "HC_emission", "PM_emission", "NOx_emission", "CO2_emission", "Fuel_level", "Oil_life", "Vibration_alert", "VibrationAlert_Total", "Vibration_Recent", "Turbo_alert", "Emission_alert", "Fog_control", "Engine_control"],');
         resultCode = pm.appendSrcLine('                                  dtype=float)');
         resultCode = pm.appendSrcLine('        prediction = _thisModelFit.predict(inputArray)');
         resultCode = pm.appendSrcLine('    except ValueError:');
         resultCode = pm.appendSrcLine('    # For models requiring or including an intercept value, a "const" column is required');
         resultCode = pm.appendSrcLine('    # For example, many statsmodels models include an intercept value that must be included for the model prediction');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[1.0, Speed_sensor, Vibration, Engine_Load, Coolant_Temp, Intake_Pressure, Engine_RPM, Speed_OBD, Intake_Air, Flow_Rate, Throttle_Pos, Voltage, Ambient, Accel, Engine_Oil_Temp, Speed_GPS, GPS_Longitude, GPS_Latitude, GPS_Bearing, GPS_Altitude, Turbo_Boost, Trip_Distance, Litres_Per_km, Accel_Ssor_Total, CO2, Trip_Time, CO_emission, HC_emission, PM_emission, NOx_emission, CO2_emission, Fuel_level, Oil_life, Vibration_alert, VibrationAlert_Total, Vibration_Recent, Turbo_alert, Emission_alert, Fog_control, Engine_control]],');
         resultCode = pm.appendSrcLine('                                columns=["const", "Speed_sensor", "Vibration", "Engine_Load", "Coolant_Temp", "Intake_Pressure", "Engine_RPM", "Speed_OBD", "Intake_Air", "Flow_Rate", "Throttle_Pos", "Voltage", "Ambient", "Accel", "Engine_Oil_Temp", "Speed_GPS", "GPS_Longitude", "GPS_Latitude", "GPS_Bearing", "GPS_Altitude", "Turbo_Boost", "Trip_Distance", "Litres_Per_km", "Accel_Ssor_Total", "CO2", "Trip_Time", "CO_emission", "HC_emission", "PM_emission", "NOx_emission", "CO2_emission", "Fuel_level", "Oil_life", "Vibration_alert", "VibrationAlert_Total", "Vibration_Recent", "Turbo_alert", "Emission_alert", "Fog_control", "Engine_control"],');
         resultCode = pm.appendSrcLine('                                dtype=float)');
         resultCode = pm.appendSrcLine('        prediction = _thisModelFit.predict(inputArray)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        EM_EVENTPROBABILITY = float(prediction)');
         resultCode = pm.appendSrcLine('    except TypeError:');
         resultCode = pm.appendSrcLine('    # If the model expects non-binary responses, a TypeError will be raised.');
         resultCode = pm.appendSrcLine('    # The except block shifts the prediction to accept a non-binary response.');
         resultCode = pm.appendSrcLine('        EM_EVENTPROBABILITY = float(prediction[:,1])');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    if (EM_EVENTPROBABILITY >= 0.2269170579029734):');
         resultCode = pm.appendSrcLine('        EM_CLASSIFICATION = "1"');
         resultCode = pm.appendSrcLine('    else:');
         resultCode = pm.appendSrcLine('        EM_CLASSIFICATION = "0" ');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)');
         revision = pm.publish(pm.getSource(), 'model_exec_02ed248b-1ff4-42ab-848c-57aa2dce04b8');

         if ( revision < 1 ) then do;
            logr.log( 'e', 'py.publish() failed.');
            resultCode = -1;
            return;
         end;
      end;
   end;
   resultCode = pm.useMethod('scoreDecisionTreeClassifier');
   if resultCode then do;
      logr.log('E', 'useMethod() failed. resultCode=$s', resultCode);
      return;
   end;
   resultCode = pm.setDouble('Speed_sensor', Speed_sensor);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Vibration', Vibration);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Engine_Load', Engine_Load);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Coolant_Temp', Coolant_Temp);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Intake_Pressure', Intake_Pressure);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Engine_RPM', Engine_RPM);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Speed_OBD', Speed_OBD);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Intake_Air', Intake_Air);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Flow_Rate', Flow_Rate);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Throttle_Pos', Throttle_Pos);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Voltage', Voltage);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Ambient', Ambient);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Accel', Accel);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Engine_Oil_Temp', Engine_Oil_Temp);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Speed_GPS', Speed_GPS);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('GPS_Longitude', GPS_Longitude);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('GPS_Latitude', GPS_Latitude);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('GPS_Bearing', GPS_Bearing);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('GPS_Altitude', GPS_Altitude);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Turbo_Boost', Turbo_Boost);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Trip_Distance', Trip_Distance);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Litres_Per_km', Litres_Per_km);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Accel_Ssor_Total', Accel_Ssor_Total);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('CO2', CO2);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Trip_Time', Trip_Time);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('CO_emission', CO_emission);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('HC_emission', HC_emission);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('PM_emission', PM_emission);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('NOx_emission', NOx_emission);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('CO2_emission', CO2_emission);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Fuel_level', Fuel_level);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Oil_life', Oil_life);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Vibration_alert', Vibration_alert);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('VibrationAlert_Total', VibrationAlert_Total);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Vibration_Recent', Vibration_Recent);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Turbo_alert', Turbo_alert);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Emission_alert', Emission_alert);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Fog_control', Fog_control);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Engine_control', Engine_control);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.execute();
   if (resultCode) then put 'Error: pm.execute failed.  resultCode=' resultCode;
   else do;
      EM_EVENTPROBABILITY = pm.getDouble('EM_EVENTPROBABILITY');
      EM_CLASSIFICATION = pm.getString('EM_CLASSIFICATION');
   end;
end;

endpackage;

