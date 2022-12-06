from sasctl.utils.pyml2ds.basic import TreeParser
from .core import EnsembleParser


class PmmlTreeParser(TreeParser):
    """Class for parsing pmml gradient boosting tree."""

    def _not_leaf(self):
        return self._node.get("defaultChild")

    def _get_var(self):
        return self._node.find("Node").find("SimplePredicate").get("field")

    def _go_left(self):
        return self._node.find("Node").get("id") == self._node.get("defaultChild")

    def _go_right(self):
        return not self._node.find("Node").get("id") == self._node.get("defaultChild")

    def _left_node(self):
        return self._node.findall("Node")[0]

    def _right_node(self):
        return self._node.findall("Node")[1]

    def _missing_node(self):
        return None

    def _split_value(self):
        return self._node.find("Node").find("SimplePredicate").get("value")

    def _decision_type(self):
        ops = {
            "lessThan": "<",
            "lessOrEqual": "<=",
            "greaterThan": ">",
            "greaterOrEqual": ">=",
        }

        return ops[self._node.find("Node").find("SimplePredicate").get("operator")]

    def _leaf_value(self):
        return self._node.get("score")


class PmmlParser(EnsembleParser):
    """Class for parsing pmml gradient boosting models.

    Parameters
    ----------
    tree_root : etree.Element
        Root node of pmml gradient boosting forest.

    """

    def __init__(self, tree_root):
        super(PmmlParser, self).__init__()

        self._tree_root = tree_root
        for elem in tree_root.iter():
            if hasattr(elem.tag, "find"):
                i = elem.tag.find("}")
                if i >= 0:
                    elem.tag = elem.tag[i + 1 :]

        self._forest = tree_root.find("MiningModel/Segmentation")[0].find("MiningModel")

        rescaleConstant = self._forest.find("Targets/Target").get("rescaleConstant")
        self.out_transform = "1 / (1 + exp(-{}))".format(
            "({0} + " + "{})".format(rescaleConstant)
        )

        self._tree_parser = PmmlTreeParser()

    def _iter_trees(self):
        for booster_id, tree_elem in enumerate(self._forest.find("Segmentation")):
            yield booster_id, tree_elem.find("TreeModel/Node")
