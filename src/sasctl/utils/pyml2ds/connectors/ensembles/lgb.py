from sasctl.utils.pyml2ds.basic import TreeParser
from .core import EnsembleParser


class LightgbmTreeParser(TreeParser):
    """Class for parsing lightgbm tree."""

    def _not_leaf(self):
        return "split_feature" in self._node

    def _get_var(self):
        return self._features[self._node["split_feature"]]

    def _go_left(self):
        return self._node["default_left"]

    def _go_right(self):
        return not self._node["default_left"]

    def _left_node(self):
        return self._node["left_child"]

    def _right_node(self):
        return self._node["right_child"]

    def _missing_node(self):
        return None

    def _split_value(self):
        return self._node["threshold"]

    def _decision_type(self):
        return self._node["decision_type"]

    def _leaf_value(self):
        return self._node["leaf_value"]


class LightgbmParser(EnsembleParser):
    """Class for parsing lightgbm model.

    Parameters
    ----------
    booster : lightgbm.basic.Booster
        Booster of lightgbm model.

    """

    def __init__(self, booster):
        super(LightgbmParser, self).__init__()

        self._booster = booster
        self._dump = booster.dump_model()

        objective = self._dump.get("objective")

        if objective != "binary sigmoid:1":
            raise ValueError(
                "Only binary sigmoid objective function is "
                "currently supported. Received '%s'." % objective
            )

        self._features = self._dump["feature_names"]
        self.out_transform = "1 / (1 + exp(-{0}))"

        self._tree_parser = LightgbmTreeParser()
        self._tree_parser._features = self._features

    def _iter_trees(self):
        for tree in self._dump["tree_info"]:
            yield tree["tree_index"], tree["tree_structure"]
