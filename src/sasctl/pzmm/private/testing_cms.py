from sasctl.pzmm import JSONFiles as JF
from sasctl import Session
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

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

model_stats = JF.calculate_model_statistics(target_value=1, validate_data=scored_df)
print(model_stats['dmcas_roc.json'])