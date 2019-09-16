import json
from .core import TreeParser


class XgbParser(TreeParser):
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
        
        self._dump = booster.get_dump(dump_format='json')
        self._objective = objective
        self._features = booster.feature_names
        
        if objective == 'binary:logistic':
            self.out_transform = "1 / (1 + exp(-{0}))"
        elif objective == 'reg:linear':
            pass
        else:
            raise Exception("Unsupported objective: %s. Supported objectives are: binary:logistic and reg:linear." % objective)


    # Generate node dictionary
    def _gen_dict(self, node):
        self.d[node['nodeid']] = node
        if 'children' in node:
            for child in node['children']:
                self._gen_dict(child)

    
    def _not_leaf(self, node):
        return 'split' in node


    def _get_var(self, node):
        return node['split']


    def _go_left(self, node):
        return (node['missing'] == node['yes'])

    
    def _go_right(self, node):
        return (node['missing'] == node['no'])


    def _left_node(self, node):
        return self.d[node['yes']]

    
    def _right_node(self, node):
        return self.d[node['no']]


    def _missing_node(self, node):
        return self.d[node['missing']]

    
    def _split_value(self, node):
        return node['split_condition']


    def _decision_type(self, node):
        return '<'


    def _leaf_value(self, node):
        return node['leaf']

    
    def _iter_trees(self):
        for booster_id, tree_json in enumerate(self._dump):
            yield booster_id, json.loads(tree_json)
