package pythonScore / overwrite=yes;
dcl package pymas pm;
dcl package logger logr('App.tk.MAS');
dcl varchar(32767) character set utf8 pypgm;
dcl int resultCode revision;

method score(double LOAN, double MORTDUE, double VALUE, varchar(100) REASON, varchar(100) JOB, double YOJ, double DEROG, double DELINQ, double CLAGE, double NINQ, double CLNO, double DEBTINC, in_out double resultCode, in_out double EM_EVENTPROBABILITY, in_out varchar(100) EM_CLASSIFICATION);
   resultCode = revision = 0;
   if null(pm) then do;
      pm = _new_ pymas();
      resultCode = pm.useModule('model_exec_0f38bb00-011c-414e-991b-822e1e1460f7', 1);
      if resultCode then do;
         resultCode = pm.appendSrcLine('import h2o');
         resultCode = pm.appendSrcLine('import gzip, shutil, os');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('import math');
         resultCode = pm.appendSrcLine('import pickle');
         resultCode = pm.appendSrcLine('import pandas as pd');
         resultCode = pm.appendSrcLine('import numpy as np');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('h2o.init()');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('_thisModelFit = h2o.load_model("/models/resources/viya/e34d30a4-66dd-4648-ad75-c6e92f0b01f1/glmFit.pickle")');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('def scoreglmFit(LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):');
         resultCode = pm.appendSrcLine('    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        global _thisModelFit');
         resultCode = pm.appendSrcLine('    except NameError:');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('        _thisModelFit = h2o.load_model("/models/resources/viya/e34d30a4-66dd-4648-ad75-c6e92f0b01f1/glmFit.pickle")');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    inputArray = pd.DataFrame([[LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],');
         resultCode = pm.appendSrcLine('                              columns=["LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"],');
         resultCode = pm.appendSrcLine('                              dtype=float, index=[0])');
         resultCode = pm.appendSrcLine('    columnTypes = {"LOAN":"numeric", "MORTDUE":"numeric", "VALUE":"numeric", "REASON":"numeric", "JOB":"numeric", "YOJ":"numeric", "DEROG":"numeric", "DELINQ":"numeric", "CLAGE":"numeric", "NINQ":"numeric", "CLNO":"numeric", "DEBTINC":"numeric"}');
         resultCode = pm.appendSrcLine('    h2oArray = h2o.H2OFrame(inputArray, column_types=columnTypes)');
         resultCode = pm.appendSrcLine('    prediction = _thisModelFit.predict(h2oArray)');
         resultCode = pm.appendSrcLine('    prediction = h2o.as_list(prediction, use_pandas=False)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    EM_EVENTPROBABILITY = float(prediction[1][2])');
         resultCode = pm.appendSrcLine('    EM_CLASSIFICATION = prediction[1][0]');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)');
         revision = pm.publish(pm.getSource(), 'model_exec_0f38bb00-011c-414e-991b-822e1e1460f7');

         if ( revision < 1 ) then do;
            logr.log( 'e', 'py.publish() failed.');
            resultCode = -1;
            return;
         end;
      end;
   end;
   resultCode = pm.useMethod('scoreglmFit');
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
   resultCode = pm.setString('REASON', REASON);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setString('JOB', JOB);
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

endpackage;

