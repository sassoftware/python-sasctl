from sasctl import Session
import sasctl.pzmm as pzmm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# load data #
df_raw = pd.read_csv('../../../../examples/data/exams.csv')

# transform data #
df = df_raw.drop(['math score', 'reading score', 'writing score'], axis=1).dropna()

df = pd.get_dummies(df, columns=df.drop('composite score', axis=1).columns)

# setting up model #
target = 'composite score'
features = df.drop(target, axis = 1).columns

# train python models #
X_train, X_test, Y_train, Y_test = train_test_split(df[features], df[target], train_size=0.7, test_size=0.3,
                                                    random_state=42)

lr = LinearRegression()
rfr = RandomForestRegressor(random_state=42)
gbr = GradientBoostingRegressor(random_state=42)

lr.fit(X_train, Y_train)
rfr.fit(X_train, Y_train)
gbr.fit(X_train, Y_train)

# create score table #
race = pd.from_dummies(X_test[['race/ethnicity_group A', 'race/ethnicity_group B', 'race/ethnicity_group C',
                               'race/ethnicity_group D', 'race/ethnicity_group E']], sep='_')
gender = pd.from_dummies(X_test[['gender_male', 'gender_female']], sep='_')
def build_score_table(model):
    score_data = {'Predicted_Composite_Score': model.predict(X_test),
                  'Composite_Score': Y_test.to_numpy(),
                  'Race': race.to_numpy()[:,0],
                  'Gender': gender.to_numpy()[:,0]}
    data = pd.DataFrame(score_data)
    return data

score_tables = {"LinearRegression": build_score_table(lr),
                "RandomForest": build_score_table(rfr),
                "GradientBoost": build_score_table(gbr)}

# running assessBias #
hostname = 'green.ingress-nginx.rint08-0020.race.sas.com'
username = 'edmdev'
password = 'Go4thsas'

sess = Session(hostname, username, password, protocol='http')

for model in ["LinearRegression", "RandomForest", "GradientBoost"]:
    pzmm.JSONFiles.assess_model_bias(
        score_table=score_tables[model],
        actual_values='Composite_Score',
        sensitive_values=['Race', 'Gender'],
        pred_values='Predicted_Composite_Score',
        json_path=Path.cwd() / f"data/BiasMetrics/examModels/{model}"
    )
