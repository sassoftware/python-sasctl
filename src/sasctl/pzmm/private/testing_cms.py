from sasctl.pzmm import JSONFiles as JF
from sasctl import Session
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json

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

# using calculate model statistics
hostname = 'green.ingress-nginx.rint08-0020.race.sas.com'
username = 'edmdev'
password = 'Go4thsas'

sess = Session(hostname, username, password, protocol='http')

# create score table
score_data = {'actual': df[target].astype(int),
              'predict': clf.predict(df_mod.drop(target, axis=1)[features]).astype(int)
              }

scored_df = pd.DataFrame(score_data)
print(scored_df)

# GETTING DATAFRAMES #


# GETTING METRICS #

# call calculate_model_stats
model_stats = JF.calculate_model_statistics(target_value=1, test_data=scored_df)

# get json files in dict form, add additional metrics
dict_fitstats = json.loads(model_stats['dmcas_fitstat.json'])['data'][-1]['dataMap']
dict_fitstats['_MISCCUTOFF_'] = dict_fitstats['_MCE_']

dict_lift = json.loads(model_stats['dmcas_lift.json'])['data'][-1]['dataMap']

dict_roc = json.loads(model_stats['dmcas_roc.json'])['data'][-1]['dataMap']
dict_roc['_TNR_'] = 1-dict_roc['_FNR_']
dict_roc['_TPR_'] = 1-dict_roc['_FPR_']

# list of all metrics (excluding _miscks_ for now)
metrics_roc = ['_KS_', '_ACC_', '_TP_', '_FP_', '_TN_', '_FN_', '_FPR_', '_FNR_', '_FDR_', '_F1_', '_TPR_', '_TNR_']
metrics_lift = ['_CumResp_', '_CumLift_', '_Lift_', '_Gain_', '_Resp_']
metrics_fitstats = ['_ASE_', '_RASE_', '_MCE_', '_MCLL_', '_KS_', '_NObs_', '_KSCut_', '_C_', '_GINI_', '_MISCCUTOFF_']

# limit dicts to only important metrics
dict_roc = {k: dict_roc[k] for k in metrics_roc if k in dict_roc}
dict_lift = {k: dict_lift[k] for k in metrics_lift if k in dict_lift}
dict_fitstats = {k: dict_fitstats[k] for k in metrics_fitstats if k in dict_fitstats}

#print(dict_roc | dict_lift | dict_fitstats)

