import xml.etree.ElementTree as etree

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

from .connectors import LightgbmParser, PmmlParser, XgbParser


def _check_type(model):
    comp_types = ["xgboost.sklearn.XGBModel", "lightgbm.LGBMModel",
                  "lightgbm.basic.Booster", "GBM.pmml file"]

    if xgboost and isinstance(model, xgboost.sklearn.XGBModel):
        if model.booster not in ['gbtree', 'dart']:
            raise RuntimeError("Model is xgboost. Unsupported booster type: %s."
                               " Supported types are: %s"
                               % (model.booster, ', '.join(comp_types)))

        parser = XgbParser(model.get_booster(), model.objective)
    elif lightgbm and isinstance(model, lightgbm.LGBMModel):
        parser = LightgbmParser(model.booster_)
    elif lightgbm and isinstance(model, lightgbm.basic.Booster):
        parser = LightgbmParser(model)
    elif etree and isinstance(model, etree.ElementTree):
        parser = PmmlParser(model.getroot())
    else:
        raise RuntimeError("Unknown booster type: %s. Compatible types are: %s."
                           " Check if corresponding library is installed."
                           % (type(model).__name__, ', '.join(comp_types)))

    return parser


def pyml2ds(in_file, out_file, out_var_name="P_TARGET"):
    """Translate a gradient boosting model and write SAS scoring code to file.

    Supported models are: xgboost, lightgbm and pmml gradient boosting.

    Parameters
    ----------
    in_file : str
        Path to file to be translated.
    out_file : str
        Path to output file with SAS code.
    out_var_name : str (optional)
        Output variable name.

    """
    # Load model file
    ext = ".pmml"
    if in_file[-len(ext):] == ext:
        model = etree.parse(in_file)
    else:
        with open(in_file, 'rb') as mf:
            model = pickle.load(mf)

    parser = _check_type(model)
    parser.out_var_name = out_var_name
    with open(out_file, "w") as f:
        parser.translate(f)
