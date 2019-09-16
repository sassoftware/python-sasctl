import sys

try:
    from lxml import etree
except ImportError:
    etree = None

try:
    import pickle
except ImportError:
    pickle = None

try:
    import xgboost
except ImportError:
    xgboost = None

try:
    import lightgbm
except ImportError:
    lightgbm = None

from .connectors import XgbParser, LightgbmParser, PmmlParser


def _check_type(model):
    comp_types = ["xgboost.sklearn.XGBModel", "lightgbm.LGBMModel", "lightgbm.basic.Booster", "GBM.pmml file"]

    parser = None
    if xgboost and isinstance(model, xgboost.sklearn.XGBModel):
        if model.booster not in ['gbtree', 'dart']:
            raise RuntimeError("Model is xgboost. Unsupported booster type: %s. Supported types are: %s" % (model.booster, ', '.join(comp_types)))

        parser = XgbParser(model.get_booster(), model.objective)
    elif lightgbm and isinstance(model, lightgbm.LGBMModel):
        parser = LightgbmParser(model.booster_)
    elif lightgbm and isinstance(model, lightgbm.basic.Booster):
        parser = LightgbmParser(model)
    elif etree and isinstance(model, etree._ElementTree):
        parser = PmmlParser(model.getroot())
    else:
        raise RuntimeError("Unknown booster type: %s. Compatible type are: %s. Check if corresponding library is installed." % type(model).__name__)

    return parser


"""Translates gradient boosting model and writes SAS scoring code to file.
Supported models are: xgboost, lightgbm and pmml gradient boosting.

Attributes
----------
inFile : str
    Path to file to be translated.
outFile : str
    Path to output file with SAS code.
outVarName : str (optional)
    Output variable name.
"""
def pyml2ds(inFile, outFile, outVarName="P_TARGET"):
    # Load model file
    ext = ".pmml"
    if inFile[-len(ext):] == ext:
        if etree is None:
            raise RuntimeError("Found pmml file but lxml is not installed. Lxml is needed to parse xml files.") 
        model = etree.parse(inFile)
    else:
        with open(inFile, 'rb') as mf:
            model = pickle.load(mf)

    parser = _check_type(model)
    parser.out_var_name = outVarName
    with open(outFile, "w") as f:
        parser.translate(f)
