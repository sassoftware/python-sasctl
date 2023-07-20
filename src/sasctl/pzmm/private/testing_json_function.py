import json
from pathlib import Path
import pandas as pd
from sasctl.pzmm import JSONFiles as JF

actual_valule = 'Survived'
prob_values = ['P_Survived1', 'P_Survived0']

groupmetrics_clf = pd.read_csv('data/example_assessBias_data_clf/gm_example_clf.csv')
maxdifferences_clf = pd.read_csv('data/example_assessBias_data_clf/md_example_clf.csv')

groupmetrics_reg = pd.read_csv('data/example_assessbias_data_reg/gm_example_reg.csv')
maxdifferences_reg = pd.read_csv('data/example_assessbias_data_reg/md_example_reg.csv')

# classification #
result = JF.bias_dataframes_to_json(
    groupmetrics=groupmetrics_clf,
    maxdifference=maxdifferences_clf,
    n_sensitivevariables= 2,
    actual_values=actual_valule,
    prob_values=prob_values,
    json_path=r'C:\Users\elmcfa\PycharmProjects\python-sasctl\src\sasctl\pzmm\private\data\clf_json_files'

)
#
#print(json.dumps(result[1], indent = 4))

# regression #
# result = JF.apply_assessbias_dataframes_to_json(
#     groupmetrics=groupmetrics_reg,
#     maxdifference=maxdifferences_reg,
#     n_sensitivevariables=2,
#     actual_values='Fare',
#     pred_values='P_Fare',
#     json_path=r'C:\Users\elmcfa\PycharmProjects\python-sasctl\src\sasctl\pzmm\private\reg_json_files'
# )

#print(json.dumps(result[0], indent=4))