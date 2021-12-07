data sasep.out;
dcl package pymas pm;
dcl package logger logr('App.tk.MAS');
dcl varchar(32767) character set utf8 pypgm;
dcl double resultCode revision;
dcl double tensor;


method score(double fixedacidity, double volatileacidity, double citricacid, double residualsugar, double chlorides, double freesulfurdioxide, double totalsulfurdioxide, double density, double pH, double sulphates, double alcohol, in_out double resultCode, in_out double tensor);
   resultCode = revision = 0;
   if null(pm) then do;
      pm = _new_ pymas();
      resultCode = pm.useModule('model_exec_3106686d-a021-449a-9c2d-f2a6f47a4948', 1);
      if resultCode then do;
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('import math');
         resultCode = pm.appendSrcLine('import pickle');
         resultCode = pm.appendSrcLine('import pandas as pd');
         resultCode = pm.appendSrcLine('import numpy as np');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('with open("/models/resources/viya/d02aadfe-618e-44e0-af6d-1bf00c6396e3/MLFlowTest.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('    _thisModelFit = pickle.load(_pFile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('def scoreMLFlowTest(fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol):');
         resultCode = pm.appendSrcLine('    "Output: tensor"');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        global _thisModelFit');
         resultCode = pm.appendSrcLine('    except NameError:');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('        with open("/models/resources/viya/d02aadfe-618e-44e0-af6d-1bf00c6396e3/MLFlowTest.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('            _thisModelFit = pickle.load(_pFile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol]],');
         resultCode = pm.appendSrcLine('                                  columns=["fixedacidity", "volatileacidity", "citricacid", "residualsugar", "chlorides", "freesulfurdioxide", "totalsulfurdioxide", "density", "pH", "sulphates", "alcohol"],');
         resultCode = pm.appendSrcLine('                                  dtype=float)');
         resultCode = pm.appendSrcLine('        prediction = _thisModelFit.predict(inputArray)');
         resultCode = pm.appendSrcLine('    except ValueError:');
         resultCode = pm.appendSrcLine('    # For models requiring or including an intercept value, a "const" column is required');
         resultCode = pm.appendSrcLine('    # For example, many statsmodels models include an intercept value that must be included for the model prediction');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[1.0, fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol]],');
         resultCode = pm.appendSrcLine('                                columns=["const", "fixedacidity", "volatileacidity", "citricacid", "residualsugar", "chlorides", "freesulfurdioxide", "totalsulfurdioxide", "density", "pH", "sulphates", "alcohol"],');
         resultCode = pm.appendSrcLine('                                dtype=float)');
         resultCode = pm.appendSrcLine('        prediction = _thisModelFit.predict(inputArray)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    tensor = prediction');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    return(tensor)');
         revision = pm.publish(pm.getSource(), 'model_exec_3106686d-a021-449a-9c2d-f2a6f47a4948');

         if ( revision < 1 ) then do;
            logr.log( 'e', 'py.publish() failed.');
            resultCode = -1;
            return;
         end;
      end;
   end;
   resultCode = pm.useMethod('scoreMLFlowTest');
   if resultCode then do;
      logr.log('E', 'useMethod() failed. resultCode=$s', resultCode);
      return;
   end;
   resultCode = pm.setDouble('fixedacidity', fixedacidity);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('volatileacidity', volatileacidity);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('citricacid', citricacid);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('residualsugar', residualsugar);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('chlorides', chlorides);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('freesulfurdioxide', freesulfurdioxide);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('totalsulfurdioxide', totalsulfurdioxide);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('density', density);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('pH', pH);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('sulphates', sulphates);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('alcohol', alcohol);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.execute();
   if (resultCode) then put 'Error: pm.execute failed.  resultCode=' resultCode;
   else do;
      tensor = pm.getDouble('tensor');
   end;
end;

method run();
    set SASEP.IN;
    score(fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol, resultCode, tensor);
end;
enddata;

