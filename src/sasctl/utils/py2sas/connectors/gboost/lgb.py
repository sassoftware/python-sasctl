from .core import TreeParser


class LightgbmParser(TreeParser):
    def __init__(self, booster):
        super(LightgbmParser, self).__init__()

        self.booster = booster
        
        self.dump = booster.dump_model()
        if self.dump['objective'] != 'binary sigmoid:1':
            raise Exception("Unfortunately only binary sigmoig objective function is supported right now. Your objective is %s. Please, open an issue at https://gitlab.sas.com/from-russia-with-love/lgb2sas." % dump['objective'])

        self.features = self.dump['feature_names']
        self.out_transform = "1 / (1 + exp(-{0}))"


    def _get_var(self, node):
        return self.features[node['split_feature']]


    # Parse lightgbm node and write SAS code to file
    def _parse_node(self, node, booster_id, f, depth=0, indent=4):
        if 'split_feature' in node:
            var = self._get_var(node)
            cond = ""
            if node['default_left']:
                cond = "missing({}) or ".format(var)
            else:
                cond = "not missing({}) and ".format(var)
            
            f.write(" " * indent * depth + "if ({}{} {} {}) then do;\n".format(cond, var, node['decision_type'], node['threshold']))
            self._parse_node(node['left_child'], booster_id, f, depth + 1)
            f.write(" " * indent * depth + "end;\n")
            f.write(" " * indent * depth + "else do;\n")
            self._parse_node(node['right_child'], booster_id, f, depth + 1)
            f.write(" " * indent * depth + "end;\n")
        else:
            f.write(" " * indent * depth + "lgbValue{} = {};\n".format(booster_id, node['leaf_value']))
    
    def iter_trees(self):
        for tree in self.dump['tree_info']:
            yield tree['tree_index'], tree['tree_structure']
