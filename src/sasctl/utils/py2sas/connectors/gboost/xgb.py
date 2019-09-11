import json
from .core import TreeParser


class XgbParser(TreeParser):
    def __init__(self, booster, objective):
        super(XgbParser, self).__init__()

        self.booster = booster
        
        self.dump = booster.get_dump(dump_format='json')
        self.objective = objective
        self.features = booster.feature_names
        
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

    
    def _get_var(self, node):
        return node['split']


    # Parse xgboost node and write SAS code to file
    def _parse_node(self, node, booster_id, f, depth=0, indent=4):
        if 'split' in node:
            var = self._get_var(node)
            cond = ""
            if (node['missing'] == node['yes']):
                cond = "missing({}) or ".format(var)
            elif (node['missing'] == node['no']):
                cond = "not missing({}) and ".format(var)
            else:
                f.write( * depth + "if (missing({})) then do;\n".format(node['split']))
                self._parse_node(self.d[node['missing']], booster_id, f, depth + 1)
                f.write(" " * indent * depth + "end;\n")
            
            f.write(" " * indent * depth + "if ({}{} < {}) then do;\n".format(cond, node['split'], node['split_condition']))
            self._parse_node(self.d[node['yes']], booster_id, f, depth + 1)
            f.write(" " * indent * depth + "end;\n")
            f.write(" " * indent * depth + "else do;\n")
            self._parse_node(self.d[node['no']], booster_id, f, depth + 1)
            f.write(" " * indent * depth + "end;\n")
        else:
            f.write(" " * indent * depth + "xgbValue{} = {};\n".format(booster_id, node['leaf']))
    
    def iter_trees(self):
        for booster_id, tree_json in enumerate(self.dump):
            yield booster_id, json.loads(tree_json)
