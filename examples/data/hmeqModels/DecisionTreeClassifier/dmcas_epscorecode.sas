data sasep.out;
dcl package pymas pm;
dcl package logger logr('App.tk.MAS');
dcl varchar(32767) character set utf8 pypgm;
dcl double resultCode revision;
dcl varchar(100) EM_CLASSIFICATION;
dcl double EM_EVENTPROBABILITY;


method score(double LOAN, double MORTDUE, double VALUE, double YOJ, double DEROG, double DELINQ, double CLAGE, double NINQ, double CLNO, double DEBTINC, in_out double resultCode, in_out double EM_EVENTPROBABILITY, in_out varchar(100) EM_CLASSIFICATION);
   resultCode = revision = 0;
   if null(pm) then do;
      pm = _new_ pymas();
      resultCode = pm.useModule('model_exec_c758b81c-ff39-4fee-a996-36af4faf29e3', 1);
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
         resultCode = pm.appendSrcLine('with open("/models/resources/viya/4b35a11a-275a-4bb9-9622-f1f3e716696f/DecisionTreeClassifier.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('    _thisModelFit = pickle.load(_pfile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('def scoreDecisionTreeClassifier(LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):');
         resultCode = pm.appendSrcLine('    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        _thisModelFit');
         resultCode = pm.appendSrcLine('    except NameError:');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('        with open("/models/resources/viya/4b35a11a-275a-4bb9-9622-f1f3e716696f/DecisionTreeClassifier.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('            _thisModelFit = pickle.load(_pFile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],');
         resultCode = pm.appendSrcLine('                                  columns=["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"],');
         resultCode = pm.appendSrcLine('                                  dtype=float)');
         resultCode = pm.appendSrcLine('        prediction = _thisModelFit.predict(inputArray)');
         resultCode = pm.appendSrcLine('    except ValueError:');
         resultCode = pm.appendSrcLine('    # For models requiring or including an intercept value, a "const" column is required');
         resultCode = pm.appendSrcLine('    # For example, many statsmodels models include an intercept value that must be included for the model prediction');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[1.0, LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],');
         resultCode = pm.appendSrcLine('                                columns=["const", "LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"],');
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
         resultCode = pm.appendSrcLine('    if (EM_EVENTPROBABILITY >= 0.199496644295302):');
         resultCode = pm.appendSrcLine('        EM_CLASSIFICATION = "1"');
         resultCode = pm.appendSrcLine('    else:');
         resultCode = pm.appendSrcLine('        EM_CLASSIFICATION = "0" ');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)');
         revision = pm.publish(pm.getSource(), 'model_exec_c758b81c-ff39-4fee-a996-36af4faf29e3');

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
   resultCode = pm.setDouble('LOAN', LOAN);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('MORTDUE', MORTDUE);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('VALUE', VALUE);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('YOJ', YOJ);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('DEROG', DEROG);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('DELINQ', DELINQ);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('CLAGE', CLAGE);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('NINQ', NINQ);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('CLNO', CLNO);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('DEBTINC', DEBTINC);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.execute();
   if (resultCode) then put 'Error: pm.execute failed.  resultCode=' resultCode;
   else do;
      EM_EVENTPROBABILITY = pm.getDouble('EM_EVENTPROBABILITY');
      EM_CLASSIFICATION = pm.getString('EM_CLASSIFICATION');
   end;
end;

method run();
    set SASEP.IN;
    score(LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC, resultCode, EM_EVENTPROBABILITY, EM_CLASSIFICATION);
end;
enddata;

