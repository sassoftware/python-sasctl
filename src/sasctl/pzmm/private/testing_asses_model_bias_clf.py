from sasctl.pzmm import JSONFiles as JF
from sasctl import Session
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Creating model/score data #

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
score_data = {'P_Survived1': clf.predict_proba(df_mod.drop(target, axis=1)[features])[:,1],
                'P_Survived0': clf.predict_proba(df_mod.drop(target, axis=1)[features])[:,0],
              'Survived': df[target],
              'Sex': df[senVar],
              'Pclass': df['Pclass']}  # original sensitive variable} # target

scored_df = pd.DataFrame(score_data)


## Examples
hostname = 'green.ingress-nginx.rint08-0020.race.sas.com'
username = 'edmdev'
password = 'Go4thsas'

sess = Session(hostname, username, password, protocol='http')


# Examples #

# only one variable passed for prob_value, multiple sensitive variables
# clf_example1 = JF.assess_model_bias(
#      score_table=scored_df,
#      actual_values='Survived',
#      prob_values='P_Survived1',
#      sensitive_values=['Sex', 'Pclass'],
#      target_level = 1
#  )

#print(f"group metrics table: {clf_example1[1].head()}")

# two variables passed for prob_value, multiple sensitive variables
# clf_example2 = JF.assess_model_bias(
#      score_table=scored_df,
#      actual_values='Survived',
#      prob_values=['P_Survived1', 'P_Survived0'],
#      sensitive_values=['Sex', 'Pclass'],
#      target_level = 1
#  )
#
# print(f"group metrics table: {clf_example2[1].head()}")

# two variables passed for prob_value, one sensitive variable
# clf_example3 = JF.assess_model_bias(
#      score_table=scored_df,
#      actual_values='Survived',
#      prob_values=['P_Survived1', 'P_Survived0'],
#      sensitive_values='Sex',
#      target_level = 1
#  )
#
# print(f"group metrics table: {clf_example3[1].head()}")

# error: did not specify target level for clf problem
# clf_example4 = JF.assess_model_bias(
#      score_table=scored_df,
#      actual_values='Survived',
#      prob_values=['P_Survived1', 'P_Survived0'],
#      sensitive_values='Sex'
#  )

# error: more than 2 elements were passed in prob_values list
# clf_example5 = JF.assess_model_bias(
#      score_table=scored_df,
#      actual_values='Survived',
#      prob_values=['P_Survived1', 'P_Survived0', 'Sex'],
#      sensitive_values='Sex',
#     target_level=1
#  )

# prob_values passed in as a list with one element
# clf_example6 = JF.assess_model_bias(
#      score_table=scored_df,
#      actual_values='Survived',
#      prob_values=['P_Survived1'],
#      sensitive_values=['Sex', 'Pclass'],
#      target_level = 1
#  )
#
# print(f"group metrics table: {clf_example6[1].head()}")

# prob_values and pred_values not passed
# clf_example7 = JF.assess_model_bias(
#      score_table=scored_df,
#      actual_values='Survived',
#      #prob_values=['P_Survived1'],
#      sensitive_values=['Sex', 'Pclass'],
#      target_level = 1
#  )
#
# print(f"group metrics table: {clf_example7[1].head()}")

# prob_values and pred_values not passed
clf_example7 = JF.assess_model_bias(
     score_table=scored_df,
     actual_values='Survived',
     prob_values=['P_Survived1'],
     sensitive_values=['Sex', 'Pclass'],
     target_level = 1
 )
#
print(f"{clf_example7[0].columns}")
#clf_example7[1].to_csv('data/gm_example.csv', index=False)