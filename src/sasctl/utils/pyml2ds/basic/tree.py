import abc
import unicodedata


class TreeParser(metaclass=abc.ABCMeta):
    """Abstract class for parsing decision tree.

    Attributes
    ----------
    d : dict
        Dictionary for storing node hierarchy. Not used in all models.
    out_transform : string
        Output transformation for generated value. For example, if logreg is
        used: 1 / (1 + exp(-{0})), where {0} stands for resulting gbvalue.
    out_var_name : string
        Name used for output variable.

    """

    def __init__(self):
        self.d = {}
        self._indent = 4
        self._depth = -1
        self._tree_id = None
        self._node = None
        self._root = None

    def init(self, root, tree_id=0):
        """Custom init method. Need to be called before using TreeParser.

        Attributes
        ----------
        root : node
            Tree root node.
        tree_id : int
            Id of current tree.

        """
        self._root = root
        self._node = root
        self._tree_id = tree_id
        self._depth = -1
        self._indent = 4

    def _gen_dict(self):
        self.d = {}

    def _get_indent(self):
        return " " * self._indent * self._depth

    @abc.abstractmethod
    def _not_leaf(self):
        pass

    @abc.abstractmethod
    def _get_var(self):
        pass

    @abc.abstractmethod
    def _go_left(self):
        pass

    @abc.abstractmethod
    def _go_right(self):
        pass

    @abc.abstractmethod
    def _left_node(self):
        pass

    @abc.abstractmethod
    def _right_node(self):
        pass

    @abc.abstractmethod
    def _missing_node(self):
        pass

    @abc.abstractmethod
    def _split_value(self):
        pass

    @abc.abstractmethod
    def _decision_type(self):
        pass

    @abc.abstractmethod
    def _leaf_value(self):
        pass

    @staticmethod
    def _remove_diacritic(input):
        output = unicodedata.normalize("NFKD", input).encode("ASCII", "ignore").decode()

        return output

    def parse_node(self, f, node=None):
        """Recursively parse tree node and write generated SAS code to file.

        Attributes
        ----------
        f : file object
            Open file for writing output SAS code.
        node: node
            Tree node to process.

        """

        pnode = self._node
        if node is not None:
            self._node = node
        else:
            self._node = self._root

        if self._node == self._root:
            self.d = {}
            self._gen_dict()

        self._depth += 1

        if self._not_leaf():
            var = self._get_var()[:32]
            var = self._remove_diacritic(var)

            split_value = self._split_value()

            cond = ""
            if self._go_left():
                cond = "missing({}) or ".format(var)
            elif self._go_right():
                cond = "not missing({}) and ".format(var)
            else:
                f.write(self._get_indent() + "if (missing(%s)) then do;\n" % var)
                self.parse_node(f, node=self._missing_node())
                f.write(self._get_indent() + "end;\n")

            f.write(
                self._get_indent()
                + "if ({}{} {} {}) then do;\n".format(
                    cond, var, self._decision_type(), split_value
                )
            )
            self.parse_node(f, node=self._left_node())
            f.write(self._get_indent() + "end;\n")
            f.write(self._get_indent() + "else do;\n")
            self.parse_node(f, node=self._right_node())
            f.write(self._get_indent() + "end;\n")
        else:
            leaf_value = self._leaf_value()

            f.write(
                self._get_indent() + "treeValue%s = %s;\n" % (self._tree_id, leaf_value)
            )

        self._node = pnode
        self._depth -= 1
