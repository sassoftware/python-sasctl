from pathlib import Path
import json
import pandas as pd

def find_file(model, fileName):
        from ..core import current_session
        from .._services.model_repository import ModelRepository as mr
        sess = current_session()
        fileList = mr.get_model_contents(model)
        for file in fileList:
            if fileName.lower() in file.name.lower():
                print(f'modelRepository/models/{model}/contents/{file.id}/content')
                correctFile = sess.get(f'modelRepository/models/{model}/contents/{file.id}/content')
                # sess.delete(f'modelRepository/{model.id}/contents/{file.id}')
                break
        return correctFile

def update_kpis(project, server="cas-shared-default", caslib="ModelPerformanceData"):
    """Updates"""
    from ..tasks import get_project_kpis
    from ..core import current_session
    from .._services.model_repository import ModelRepository as mr
    from io import StringIO
    sess = current_session()
    kpis = get_project_kpis(project, server, caslib)
    modelsToUpdate = kpis['ModelUUID'].unique().tolist()
    for model in modelsToUpdate:
        currentParams = find_file(model, "Hyperparameters")
        currentJSON = currentParams.json()
        modelRows = kpis.loc[kpis['ModelUUID'] == model]
        modelRows.set_index("TimeLabel", inplace=True)
        kpiJSON = modelRows.to_json(orient='index')
        parsedJSON = json.loads(kpiJSON)
        currentJSON['kpis'] = parsedJSON
        # print(currentJSON)
        # sess.post(f"modelRepository/models/{model}/contents", files={f"{currentJSON['kpis']['ModelName']}test.json": StringIO(json.dumps(currentJSON, indent=4))}, data={"name": f"{currentJSON['kpis']['ModelName']}test.json"})
        mr.add_model_content(model, StringIO((json.dumps(currentJSON, indent=4))), f"{currentJSON['kpis'][list(currentJSON['kpis'].keys())[0]]['ModelName']}hyperparameters.json")
                
def sklearn_params(model, modelPrefix, pPath):
    hyperparameters = model.get_params()
    modelJson = {"hyperparameters": hyperparameters}
    with open(Path(pPath) / (f"{modelPrefix}Hyperparameters.json"), "w") as f:
        f.write(json.dumps(modelJson))

class modelParameters:
    def generate_hyperparameters(model, modelPrefix, pPath):
        if all(hasattr(model, attr) for attr in ["_estimator_type", "get_params"]):
            sklearn_params(model, modelPrefix, pPath)
        else:
            print("Other model types not currently supported.")
