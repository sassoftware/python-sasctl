from pyml2ds.basic.tree import TreeParser
from .core import EnsembleParser


class ForestTreeParser(TreeParser):
    """Class for parsing random forest tree.=
    """

    """Custom init method. Need to be called before using TreeParser.

    Attributes
    ----------
    root : sklearn.tree.DecisionTreeClassifier
        Skleran tree classifier.
    tree_id : int
        Id of current tree.
    """
    def init(self, root, tree_id=0):
        super(ForestTreeParser, self).init(0)

        self._tree = root.tree_


    def _not_leaf(self):
        return self._left_node() != self._right_node()


    def _get_var(self):
        return "feature%d" % self._tree.feature[self._node]


    def _go_left(self):
        return True

    
    def _go_right(self):
        return False


    def _left_node(self):
        return self._tree.children_left[self._node]

    
    def _right_node(self):
        return self._tree.children_right[self._node]


    def _missing_node(self):
        return None

    
    def _split_value(self):
        return self._tree.threshold[self._node]


    def _decision_type(self):
        return '<='


    def _leaf_value(self):
        value = self._tree.value[self._node]
        return (value / value.sum())[0,1]


class ForestParser(EnsembleParser):
    """Class for parsing random forest model.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        Forest model.
    """
    def __init__(self, model):
        super(ForestParser, self).__init__()

        self._model = model
        
        #self._features = self._dump['feature_names']
        self.out_transform = "{0}"
        self._tree_parser = ForestTreeParser()


    def _iter_trees(self):
        for i, tree in enumerate(self._model.estimators_):
            yield i, tree

    
    def _aggregate(self, booster_count):
        return "treeValue = sum({}) / %d;\n".format(', '.join(["treeValue%d" % i for i in range(booster_count)]), booster_count)
