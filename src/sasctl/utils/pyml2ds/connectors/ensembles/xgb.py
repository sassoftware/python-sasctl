import json

from sasctl.utils.pyml2ds.basic import TreeParser
from .core import EnsembleParser


class XgbTreeParser(TreeParser):
    """Class for parsing xgboost tree."""

    # Generate node dictionary
    def _gen_dict(self):
        pnode = self._node

        self.d[self._node["nodeid"]] = self._node
        if "children" in self._node:
            for child in self._node["children"]:
                self._node = child
                self._gen_dict()

        self._node = pnode

    def _not_leaf(self):
        return "split" in self._node

    def _get_var(self):
        return self._node["split"]

    def _go_left(self):
        return self._node["missing"] == self._node["yes"]

    def _go_right(self):
        return self._node["missing"] == self._node["no"]

    def _left_node(self):
        return self.d[self._node["yes"]]

    def _right_node(self):
        return self.d[self._node["no"]]

    def _missing_node(self):
        return self.d[self._node["missing"]]

    def _split_value(self):
        return self._node["split_condition"]

    def _decision_type(self):
        return "<"

    def _leaf_value(self):
        return self._node["leaf"]


class XgbParser(EnsembleParser):
    """Class for parsing xgboost model.

    Parameters
    ----------
    booster : xgboost.core.Booster
        Booster of xgboost model.
    objective : {'reg:linear', 'binary:logistic'}
        Xgboost objective function.

    """

    def __init__(self, booster, objective):
        super(XgbParser, self).__init__()

        self._booster = booster
        self._dump = booster.get_dump(dump_format="json")
        self._objective = objective
        self._features = booster.feature_names

        if objective == "binary:logistic":
            self.out_transform = "1 / (1 + exp(-{0}))"
        elif objective == "reg:linear":
            pass
        else:
            raise ValueError(
                "Unsupported objective: '%s'.  "
                "Expected "
                "'binary:logistic' or 'reg:linear'." % objective
            )

        self._tree_parser = XgbTreeParser()

    def _iter_trees(self):
        for booster_id, tree_json in enumerate(self._dump):
            yield booster_id, json.loads(tree_json)
