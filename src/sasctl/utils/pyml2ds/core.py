import os
import pickle
import xml.etree.ElementTree as etree
from io import StringIO

try:
    import xgboost
except ImportError:
    xgboost = None

try:
    import lightgbm
except ImportError:
    lightgbm = None

from sasctl.utils.decorators import experimental
from .connectors import LightgbmParser, PmmlParser, XgbParser


def _check_type(model):
    comp_types = [
        "xgboost.sklearn.XGBModel",
        "lightgbm.LGBMModel",
        "lightgbm.basic.Booster",
        "GBM.pmml file",
    ]

    if xgboost and isinstance(model, xgboost.sklearn.XGBModel):
        if model.booster not in ["gbtree", "dart"]:
            raise RuntimeError(
                "Model is xgboost. Unsupported booster type: %s."
                " Supported types are: %s" % (model.booster, ", ".join(comp_types))
            )

        parser = XgbParser(model.get_booster(), model.objective)
    elif lightgbm and isinstance(model, lightgbm.LGBMModel):
        parser = LightgbmParser(model.booster_)
    elif lightgbm and isinstance(model, lightgbm.basic.Booster):
        parser = LightgbmParser(model)
    elif etree and isinstance(model, etree.ElementTree):
        parser = PmmlParser(model.getroot())
    else:
        raise RuntimeError(
            "Unknown booster type: %s. Compatible types are: %s."
            " Check if corresponding library is installed."
            % (type(model).__name__, ", ".join(comp_types))
        )

    return parser


@experimental
def pyml2ds(in_file, out_var_name="P_TARGET"):
    """Translate a gradient boosting model and write SAS scoring code to file.

    Supported models are: xgboost, lightgbm and pmml gradient boosting.

    Parameters
    ----------
    in_file : str or bytes or file-like
        Pickled object to translate.  String is assumed to be a path to a picked
        file, file-like is assumed to be an open file handle to a pickle
        object, and bytes is assumed to be the raw pickled bytes.
    out_var_name : str (optional)
        Output variable name.

    Returns
    -------
    str
        A SAS Data Step program implementing the model.

    Examples
    --------
    Generate SAS code from an XGBoost model.

    >>> from xgboost.sklearn import XGBRegressor
    >>> xgb = XGBRegressor()
    >>> xgb.fit(X, y)
    >>> pkl = pickle.dumps(xgb)
    >>> sas_code = pyml2ds(pkl)

    """

    try:
        # In Python2 str could either be a path or the binary pickle data,
        # so check if its a valid filepath too.
        is_file_path = isinstance(in_file, str) and os.path.isfile(in_file)
    except TypeError:
        is_file_path = False

    # Path to a PMML or pickle file
    if is_file_path:
        # Parse PMML files
        if os.path.splitext(in_file)[-1] == ".pmml":
            model = etree.parse(in_file)
        else:
            # Read pickled files
            with open(in_file, "rb") as f:
                model = pickle.load(f)

    elif isinstance(in_file, bytes):
        # Assume byte string is the actual pickled bytes
        model = pickle.loads(in_file)
    else:
        # Assume a file object containing the pickled object
        model = pickle.load(in_file)

    # Verify model is a valid type
    parser = _check_type(model)
    parser.out_var_name = out_var_name

    # Parser is currently written to expect a file input
    # Until refactored, use StringIO to collect the text in memory
    f = StringIO()
    parser.translate(f)

    # Return contents of "file"
    f.seek(0)
    return f.read()
