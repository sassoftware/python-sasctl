from sasctl import Session
import sasctl.pzmm as pzmm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Creating model/score data #

# load data #
df_raw = pd.read_csv('data/titanic.csv')

# transform data
columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = df_raw.dropna(subset=columns)
df['Survived'] = df['Survived'].astype(str)
df = pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked'])

# set up model #
features = ['Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3',
            'Embarked_C', 'Embarked_Q', 'Embarked_S']
target = 'Survived'
senVar = 'Sex'

# train python classification models #
X_train, X_test, Y_train, Y_test = train_test_split(df[features], df[target], train_size=0.7, test_size=0.3,
                                                    random_state=42)

dtc = DecisionTreeClassifier(random_state=42)
rfc = RandomForestClassifier(random_state=42)
gbc = GradientBoostingClassifier(random_state=42)

dtc.fit(X_train, Y_train)
rfc.fit(X_train, Y_train)
gbc.fit(X_train, Y_train)

# create score table #
sex = pd.from_dummies(X_test[['Sex_male', 'Sex_female']], sep='_')
def build_score_table(model):
    score_data = {'P_Survived1': model.predict_proba(X_test)[:,1],
                  'P_Survived0': model.predict_proba(X_test)[:, 0],
                  'Survived': Y_test.to_numpy(),
                  'Sex': sex.to_numpy()[:,0]}
    data = pd.DataFrame(score_data)
    return data

score_tables = {"DecisionTree": build_score_table(dtc),
                "RandomForest": build_score_table(rfc),
                "GradientBoost": build_score_table(gbc)}

# running assessBias #
hostname = 'green.ingress-nginx.rint08-0020.race.sas.com'
username = 'edmdev'
password = 'Go4thsas'

sess = Session(hostname, username, password, protocol='http')

for i, model in enumerate(["DecisionTree", "RandomForest", "GradientBoost"]):
    pzmm.JSONFiles.assess_model_bias(
        score_table=score_tables[model],
        actual_values='Survived',
        sensitive_values='Sex',
        prob_values=['P_Survived1', 'P_Survived0'],
        json_path=fr"C:\Users\elmcfa\PycharmProjects\python-sasctl\examples\data\BiasMetrics\titanicModels\{model}"

    )



