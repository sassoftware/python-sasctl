import json
from pathlib import Path

import pandas as pd

from sasctl.pzmm import JSONFiles as JF

actual_valule = "Survived"
prob_values = ["P_Survived1", "P_Survived0"]

data = pd.read_csv("data/gm_example.csv")
path = "template_files\clf_jsons\groupMetrics.json"

json_dict = [{}, {}]
json_dict[1] = JF.read_json_file(path)

# updating data rows
for row_num in range(len(data)):
    row_dict = data.iloc[row_num].replace(float("nan"), None).to_dict()
    new_data = {"dataMap": row_dict, "rowNumber": row_num + 1}
    json_dict[1]["data"].append(new_data)

# first is target
for i, label in enumerate(reversed(prob_values)):
    json_dict[1]["parameterMap"][f"predict_proba{i}"][
        "label"
    ] = f"Average Predicted: {actual_valule}={i}"
    json_dict[1]["parameterMap"][f"predict_proba{i}"]["parameter"] = label
    json_dict[1]["parameterMap"][f"predict_proba{i}"]["values"] = [label]

    json_dict[1]["parameterMap"] = {
        label if k == f"predict_proba{i}" else k: v
        for k, v in json_dict[1]["parameterMap"].items()
    }

json_file = json.dumps(json_dict[1], indent=4)

print(json_file)
