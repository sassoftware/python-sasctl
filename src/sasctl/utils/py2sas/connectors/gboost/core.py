import json


class TreeParser:
    def __init__(self, out_transform="{0}", out_var_name="P_TARGET"):
        self.d = {}
        self.out_transform = out_transform
        self.out_var_name = out_var_name


    def _gen_dict(self, node):
        pass


    def _parse_node(self, node, booster_id, f, depth=0, indent=4):
        pass


    def iter_trees(self):
        pass


    def translate(self, f):
        for booster_id, tree in self.iter_trees():
            f.write("/* Parsing tree {}*/\n".format(booster_id))
            
            self.d = dict()
            self._gen_dict(tree)
            self._parse_node(tree, booster_id, f)
            
            f.write("\n")
        
        f.write("/* Getting target probability */\n")
        f.write("xgbValue = sum({});\n".format(', '.join(["xgbValue" + str(i) for i in range(booster_id + 1)])))
        f.write("{} = {};\n".format(self.out_var_name, self.out_transform.format("xgbValue")))
