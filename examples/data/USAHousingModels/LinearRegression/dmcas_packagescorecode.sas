package pythonScore / overwrite=yes;
dcl package pymas pm;
dcl package logger logr('App.tk.MAS');
dcl varchar(32767) character set utf8 pypgm;
dcl int resultCode revision;

method score(double Avg_Area_Income, double Avg_Area_House_Age, double Avg_Area_Number_of_Rooms, double Avg_Area_Number_of_Bedrooms, double Area_Population, in_out double resultCode, in_out double EM_PREDICTION, in_out double EM_PREDICTION);
   resultCode = revision = 0;
   if null(pm) then do;
      pm = _new_ pymas();
      resultCode = pm.useModule('model_exec_5b58f10f-2fde-4de7-a36e-96618a770c71', 1);
      if resultCode then do;
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('import math');
         resultCode = pm.appendSrcLine('import pickle');
         resultCode = pm.appendSrcLine('import pandas as pd');
         resultCode = pm.appendSrcLine('import numpy as np');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('with open("/models/resources/viya/cb485d04-6cb9-48ce-a829-67b8167303ce/LinearRegression.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('    _thisModelFit = pickle.load(_pFile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('def scoreLinearRegression(Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population):');
         resultCode = pm.appendSrcLine('    "Output: EM_PREDICTION, EM_PREDICTION"');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        global _thisModelFit');
         resultCode = pm.appendSrcLine('    except NameError:');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('        with open("/models/resources/viya/cb485d04-6cb9-48ce-a829-67b8167303ce/LinearRegression.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('            _thisModelFit = pickle.load(_pFile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population]],');
         resultCode = pm.appendSrcLine('                                  columns=["Avg_Area_Income", "Avg_Area_House_Age", "Avg_Area_Number_of_Rooms", "Avg_Area_Number_of_Bedrooms", "Area_Population"],');
         resultCode = pm.appendSrcLine('                                  dtype=float)');
         resultCode = pm.appendSrcLine('        prediction = _thisModelFit.predict(inputArray)');
         resultCode = pm.appendSrcLine('    except ValueError:');
         resultCode = pm.appendSrcLine('    # For models requiring or including an intercept value, a "const" column is required');
         resultCode = pm.appendSrcLine('    # For example, many statsmodels models include an intercept value that must be included for the model prediction');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[1.0, Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population]],');
         resultCode = pm.appendSrcLine('                                columns=["const", "Avg_Area_Income", "Avg_Area_House_Age", "Avg_Area_Number_of_Rooms", "Avg_Area_Number_of_Bedrooms", "Area_Population"],');
         resultCode = pm.appendSrcLine('                                dtype=float)');
         resultCode = pm.appendSrcLine('        prediction = _thisModelFit.predict(inputArray)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        EM_PREDICTION = float(prediction)');
         resultCode = pm.appendSrcLine('    except TypeError:');
         resultCode = pm.appendSrcLine('    # If the model expects non-binary responses, a TypeError will be raised.');
         resultCode = pm.appendSrcLine('    # The except block shifts the prediction to accept a non-binary response.');
         resultCode = pm.appendSrcLine('        EM_PREDICTION = float(prediction[:,1])');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    if (EM_PREDICTION >= 1232072.6541453):');
         resultCode = pm.appendSrcLine('        EM_PREDICTION = "1"');
         resultCode = pm.appendSrcLine('    else:');
         resultCode = pm.appendSrcLine('        EM_PREDICTION = "0" ');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    return(EM_PREDICTION, EM_PREDICTION)');
         revision = pm.publish(pm.getSource(), 'model_exec_5b58f10f-2fde-4de7-a36e-96618a770c71');

         if ( revision < 1 ) then do;
            logr.log( 'e', 'py.publish() failed.');
            resultCode = -1;
            return;
         end;
      end;
   end;
   resultCode = pm.useMethod('scoreLinearRegression');
   if resultCode then do;
      logr.log('E', 'useMethod() failed. resultCode=$s', resultCode);
      return;
   end;
   resultCode = pm.setDouble('Avg_Area_Income', Avg_Area_Income);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Avg_Area_House_Age', Avg_Area_House_Age);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Avg_Area_Number_of_Rooms', Avg_Area_Number_of_Rooms);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Avg_Area_Number_of_Bedrooms', Avg_Area_Number_of_Bedrooms);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('Area_Population', Area_Population);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.execute();
   if (resultCode) then put 'Error: pm.execute failed.  resultCode=' resultCode;
   else do;
      EM_PREDICTION = pm.getDouble('EM_PREDICTION');
      EM_PREDICTION = pm.getDouble('EM_PREDICTION');
   end;
end;

endpackage;

