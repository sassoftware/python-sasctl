from sasctl.pzmm import JSONFiles as JF
from sasctl import Session
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

hostname = 'green.ingress-nginx.rint08-0020.race.sas.com'
username = 'edmdev'
password = 'Go4thsas'

sess = Session(hostname, username, password, protocol='http')

df = pd.read_csv('data/titanic.csv')

# transform data for model
columns = ['Survived', 'Pclass', 'Sex', 'Age',
           'SibSp', 'Parch', 'Fare', 'Embarked']
df = df.dropna(subset=columns)
df_mod = pd.get_dummies(df, columns=['Survived', 'Pclass', 'Sex', 'Embarked'])

# set up model
target = 'Fare'
features = ['Age', 'SibSp', 'Parch', 'Survived_0', 'Survived_1',
            'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
            'Embarked_C', 'Embarked_S']
senVar = 'Sex'

# train python model, linear regression
X_train, X_test, Y_train, Y_test = train_test_split(df_mod[features], df_mod[target], train_size=0.8, test_size=0.2,
                                                    random_state=123)

reg = LinearRegression()
reg.fit(X_train, Y_train)

# create score table
score_data = {'P_Fare': reg.predict(df_mod.drop(target, axis=1)[features]),  # using all data
              'Sex': df[senVar],  # get og column of Pclass
              'Fare': df[target],
              'Pclass': df.Pclass}

scored_df = pd.DataFrame(score_data)

output = JF.calculate_group_metrics(
            score_table= scored_df,
            actual_value= 'Fare',
            pred_value= 'P_Fare',
            sensitive_value= 'Sex'
)

print(output.head())