from sasctl.pzmm import JSONFiles as JF
from sasctl import Session
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Creating model/score data #

# load data
df = pd.read_csv('data/hmeq.csv')

# transform data
df = df.dropna()

# set up model
features = ['BAD', 'LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG',
       'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
target = 'JOB'

# train python model, gradient boost decision tree
X_train, X_test, Y_train, Y_test = train_test_split(df[features], df[target], train_size=0.8, test_size=0.2,
                                                    random_state=123)
clf = RandomForestClassifier(random_state=123)
clf.fit(X_train, Y_train)

# create score table
score_data = {'P_JobMgr': clf.predict_proba(X_test)[:,0],
              'P_JobOffice': clf.predict_proba(X_test)[:,1],
              'P_JobOther': clf.predict_proba(X_test)[:,2],
              'P_JobProfExe': clf.predict_proba(X_test)[:,3],
              'P_JobSales': clf.predict_proba(X_test)[:,4],
              'P_JobSelf': clf.predict_proba(X_test)[:,5],
              'JOB': Y_test,
              'BAD': X_test['BAD'].astype(str)}

scored_df = pd.DataFrame(score_data)

## Examples
hostname = 'green.ingress-nginx.rint08-0020.race.sas.com'
username = 'edmdev'
password = 'Go4thsas'

sess = Session(hostname, username, password, protocol='http')
multiclass1 = JF.assess_model_bias(
     score_table=scored_df,
     actual_values='JOB',
     prob_values=['P_JobMgr', 'P_JobOffice', 'P_JobOther', 'P_JobProfExe', 'P_JobSales', 'P_JobSelf'],
     sensitive_values=['BAD'],
    levels=['Mgr', 'Office', 'Other', 'ProfExe', 'Sales', 'Self'],
    return_dataframes=True
 )
#
# # the first probability class will be the predicted event class
#
print(multiclass1["maxDifferencesData"])

