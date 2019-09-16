import abc, six


@six.add_metaclass(abc.ABCMeta)
class TreeParser:
    """Abstract class for parsing gbm forest.

    Attributes
    ----------
    d : dict
        Dictionary for storing node hierarchy. Used not in all models. 
    out_transform : string
        Output transformation for generated value. For example, for logreg is used: 1 / (1 + exp(-{0})), where {0} stands for resulting gbvalue.
    out_var_name : string
        Name used for output variable.
    """
    def __init__(self, out_transform="{0}", out_var_name="P_TARGET"):
        self.d = {}
        self.out_transform = out_transform
        self.out_var_name = out_var_name


    def _gen_dict(self, node):
        self.d = {}


    @abc.abstractmethod
    def _not_leaf(self, node):
        pass


    @abc.abstractmethod
    def _get_var(self, node):
        pass


    @abc.abstractmethod
    def _go_left(self, node):
        pass


    @abc.abstractmethod
    def _go_right(self, node):
        pass


    @abc.abstractmethod
    def _left_node(self, node):
        pass


    @abc.abstractmethod
    def _right_node(self, node):
        pass


    @abc.abstractmethod
    def _missing_node(self, node):
        pass


    @abc.abstractmethod
    def _split_value(self, node):
        pass


    @abc.abstractmethod
    def _decision_type(self, node):
        pass


    @abc.abstractmethod
    def _leaf_value(self, node):
        pass


    @abc.abstractmethod
    def _iter_trees(self):
        pass


    def _parse_node(self, node, booster_id, f, depth=0, indent=4):
        if self._not_leaf(node):
            var = self._get_var(node)[:32]
            cond = ""
            if self._go_left(node):
                cond = "missing({}) or ".format(var)
            elif self._go_right(node):
                cond = "not missing({}) and ".format(var)
            else:
                f.write( * depth + "if (missing({})) then do;\n".format(var))
                self._parse_node(self._missing_node(node), booster_id, f, depth + 1)
                f.write(" " * indent * depth + "end;\n")
            
            f.write(" " * indent * depth + "if ({}{} {} {}) then do;\n".format(cond, var, self._decision_type(node), self._split_value(node)))
            self._parse_node(self._left_node(node), booster_id, f, depth + 1)
            f.write(" " * indent * depth + "end;\n")
            f.write(" " * indent * depth + "else do;\n")
            self._parse_node(self._right_node(node), booster_id, f, depth + 1)
            f.write(" " * indent * depth + "end;\n")
        else:
            f.write(" " * indent * depth + "gbValue{} = {};\n".format(booster_id, self._leaf_value(node)))


    """Translates gradient boosting model and writes SAS scoring code to file.

    Attributes
    ----------
    f : file object
        Open file for writing output SAS code.
    """
    def translate(self, f):
        for booster_id, tree in self._iter_trees():
            f.write("/* Parsing tree {}*/\n".format(booster_id))
            
            self.d = dict()
            self._gen_dict(tree)
            self._parse_node(tree, booster_id, f)
            
            f.write("\n")
        
        f.write("/* Getting target probability */\n")
        f.write("gbValue = sum({});\n".format(', '.join(["gbValue" + str(i) for i in range(booster_id + 1)])))
        f.write("{} = {};\n".format(self.out_var_name, self.out_transform.format("gbValue")))
