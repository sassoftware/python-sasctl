from pathlib import Path
import json

from .._services.model_repository import ModelRepository as mr


def find_file(model, fileName):
    from ..core import current_session

    sess = current_session()
    fileList = mr.get_model_contents(model)
    for file in fileList:
        if fileName.lower() in file.name.lower():
            correctFile = sess.get(
                "modelRepository/models/{}/contents/{}/content".format(model, file.id)
            )
            break
    return correctFile


def sklearn_params(model, modelPrefix, pPath):
    hyperparameters = model.get_params()
    modelJson = {"hyperparameters": hyperparameters}
    with open(Path(pPath) / ("{}Hyperparameters.json".format(modelPrefix)), "w") as f:
        f.write(json.dumps(modelJson, indent=4))


class modelParameters:
    def generate_hyperparameters(model, modelPrefix, pPath):
        if all(hasattr(model, attr) for attr in ["_estimator_type", "get_params"]):
            sklearn_params(model, modelPrefix, pPath)
        else:
            print(
                "Other model types not currently supported for hyperparameter generation."
            )

    def update_kpis(
        project,
        server="cas-shared-default",
        caslib="ModelPerformanceData",
    ):
        """Updates"""
        from ..tasks import get_project_kpis
        from io import StringIO

        kpis = get_project_kpis(project, server, caslib)
        modelsToUpdate = kpis["ModelUUID"].unique().tolist()
        for model in modelsToUpdate:
            currentParams = find_file(model, "hyperparameters")
            currentJSON = currentParams.json()
            modelRows = kpis.loc[kpis["ModelUUID"] == model]
            modelRows.set_index("TimeLabel", inplace=True)
            kpiJSON = modelRows.to_json(orient="index")
            parsedJSON = json.loads(kpiJSON)
            currentJSON["kpis"] = parsedJSON
            fileName = "{}Hyperparameters.json".format(
                currentJSON["kpis"][list(currentJSON["kpis"].keys())[0]]["ModelName"]
            )
            mr.add_model_content(
                model,
                StringIO((json.dumps(currentJSON, indent=4))),
                fileName,
            )

    @classmethod
    def get_hyperparameters(model):
        return find_file(model).json()
