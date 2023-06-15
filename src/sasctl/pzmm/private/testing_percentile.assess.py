import swat
from sasctl import Session
from sasctl.pzmm import JSONFiles as JF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

hostname = 'green.ingress-nginx.rint08-0020.race.sas.com'
username = 'edmdev'
password = 'Go4thsas'

sess = Session(hostname, username, password, protocol='http')


# load data
df = pd.read_csv('data/titanic.csv')

# transform data
columns = ['Survived', 'Pclass', 'Sex', 'Age',
           'SibSp', 'Parch', 'Fare', 'Embarked']

df = df.dropna(subset=columns)
df['Survived'] = df['Survived'].astype(str)
df_mod = pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked'])  # modified df for model

# set up model
features = ['Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male',
            'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q',
            'Embarked_S']
target = 'Survived'
senVar = 'Sex'

# train python model, gradient boost decision tree
X_train, X_test, Y_train, Y_test = train_test_split(df_mod[features], df_mod[target], train_size=0.8, test_size=0.2,
                                                    random_state=123)
clf = RandomForestClassifier(random_state=123)
clf.fit(X_train, Y_train)

# create score table
score_data = {'Survived': df[target],
              'Probability': clf.predict_proba(df_mod.drop(target, axis=1)[features])[:,1],
              'P_Survived': clf.predict(df_mod.drop(target, axis=1)[features]),
              'Sex': df[senVar]
              }

scored_df = pd.DataFrame(score_data)


ouput = JF.calculate_group_metrics(
            score_table= scored_df,
            actual_value= 'Survived',
            pred_value= 'Probability',
            target_level= 1,
            sensitive_value= 'Sex',
            type= "class"
)

print(ouput[['INTO_EVENT', 'LEVEL', 'PREDICTED_EVENT', 'P_Survived0', 'P_Survived1']])



# for level in list(df[senVar].unique()):
#     sub_df = scored_df[scored_df[senVar] == level][['P_Survived', 'Survived']]
#     sub_df = sub_df.astype(int)
#     #print(sub_df.head())
#
#     # transform score table to correct df for percentile.assess
#     data = JF.stat_dataset_to_dataframe(sub_df, '1')
#     print(data.head())
#
#     # using calculate model statistics
#     hostname = 'green.ingress-nginx.rint08-0020.race.sas.com'
#     username = 'edmdev'
#     password = 'Go4thsas'
#
#     sess = Session(hostname, username, password, protocol='http')
#     conn = sess.as_swat()
#
#     conn.loadactionset(actionset="percentile")
#
#     conn.upload(data, casout={"name": "assess_dataset", "replace": True, "caslib": "Public"})
#
#     conn.percentile.assess(
#                     table={"name": "assess_dataset", "caslib": "Public"},
#                     response="predict",
#                     pVar="predict_proba",
#                     event='1',
#                     pEvent='0.5',
#                     inputs="actual",
#                     nBins=20,
#                     fitStatOut={"name": "FitStat", "replace": True, "caslib": "Public"},
#                     rocOut={"name": "ROC", "replace": True, "caslib": "Public"},
#                     casout={"name": "Lift", "replace": True, "caslib": "Public"},
#                 )
#
#     # get individual dataframes
#     fitstat_df = pd.DataFrame(conn.CASTable("FitStat", caslib="Public").to_frame()).reset_index()
#     roc_df = pd.DataFrame(conn.CASTable("ROC", caslib="Public").to_frame()).iloc[[-1]].reset_index()
#     lift_df = pd.DataFrame(conn.CASTable("Lift", caslib="Public").to_frame()).iloc[[-1]].reset_index()
#
#     # make new dataframe with all relevant info
#     fitstat_df = fitstat_df[['_ASE_', '_RASE_', '_MCE_', '_MCLL_', '_NObs_']]
#     fitstat_df['_MISCCUTOFF_'] = fitstat_df['_MCE_']
#
#     roc_df = roc_df[['_KS_', '_ACC_', '_TP_', '_FP_', '_TN_', '_FN_', '_FPR_', '_FNR_', '_FDR_', '_F1_', '_GINI_', '_C_']]
#     roc_df['_TPR_'] = 1-roc_df['_FPR_']
#     roc_df['_TNR_'] = 1-roc_df['_FNR_']
#
#     lift_df = lift_df[['_CumResp_', '_CumLift_', '_Lift_', '_Gain_', '_Resp_']]
#
#     result = pd.concat([fitstat_df, roc_df, lift_df], axis=1)
#
#     result['VLABEL'] = senVar
#     result['Level'] = level
#
#     print(result.iloc[0])



