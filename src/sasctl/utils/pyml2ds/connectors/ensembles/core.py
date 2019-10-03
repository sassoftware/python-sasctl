import abc, six


@six.add_metaclass(abc.ABCMeta)
class EnsembleParser:
    """Abstract class for parsing decision tree ensebmles.

    Attributes
    ----------
    out_transform : string
        Output transformation for generated value. For example, for logreg is used: 1 / (1 + exp(-{0})), where {0} stands for resulting gbvalue.
    out_var_name : string
        Name used for output variable.
    """
    def __init__(self, out_transform="{0}", out_var_name="P_TARGET"):
        self.out_transform = out_transform
        self.out_var_name = out_var_name


    @abc.abstractmethod
    def _iter_trees(self):
        pass

    
    def _aggregate(self, booster_count):
        return "treeValue = sum({});\n".format(', '.join(["treeValue%d" % i for i in range(booster_count)]))


    """Translates gradient boosting model and writes SAS scoring code to file.

    Attributes
    ----------
    f : file object
        Open file for writing output SAS code.
    """
    def translate(self, f, test=False):
        for booster_id, tree in self._iter_trees():
            f.write("/* Parsing tree {}*/\n".format(booster_id))
            
            self._tree_parser.init(tree, booster_id)
            self._tree_parser.parse_node(f, test=test)
            
            f.write("\n")
        
        f.write("/* Getting target probability */\n")
        f.write(self._aggregate(booster_id + 1))
        f.write("{} = {};\n".format(self.out_var_name, self.out_transform.format("treeValue")))
