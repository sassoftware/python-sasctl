import sys
import pickle
import xgboost
import lightgbm
from .connectors import XgbParser, LightgbmParser


def _check_type(model):
    comp_types = ["xgboost.sklearn.XGBModel", "lightgbm.LGBMModel", "lightgbm.basic.Booster"]

    parser = None
    if isinstance(model, xgboost.sklearn.XGBModel):
        if model.booster not in ['gbtree', 'dart']:
            raise Exception("Model is xgboost. Unsupported booster type: %s. Supported types are: gbtree, dart." % model.booster)

        parser = XgbParser(model.get_booster(), model.objective)
    elif isinstance(model, lightgbm.LGBMModel):
        parser = LightgbmParser(model.booster_)
    elif isinstance(model, lightgbm.basic.Booster):
        parser = LightgbmParser(model)
    else:
        raise Exception("Unknown xgb type: %s. Compatible type are: %s." % type(model).__name__)

    return parser


def py2sas(inFile, outFile, outVarName="P_TARGET"):
    # Load model file
    with open(inFile, 'rb') as mf:
        model = pickle.load(mf)

    parser = _check_type(model)
    with open(outFile, "w") as f:
        parser.translate(f)
