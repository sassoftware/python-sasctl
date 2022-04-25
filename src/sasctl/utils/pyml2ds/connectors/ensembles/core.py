import abc


class EnsembleParser(metaclass=abc.ABCMeta):
    """Abstract class for parsing decision tree ensembles.

    Attributes
    ----------
    out_transform : string
        Output transformation for generated value. For example, if logreg is
        used: 1 / (1 + exp(-{0})), where {0} stands for resulting gbvalue.
    out_var_name : string
        Name used for output variable.

    """

    def __init__(self, out_transform="{0}", out_var_name="P_TARGET"):
        self.out_transform = out_transform
        self.out_var_name = out_var_name

    @abc.abstractmethod
    def _iter_trees(self):
        pass

    @classmethod
    def _aggregate(cls, booster_count):
        return "treeValue = sum({});\n".format(
            ", ".join("treeValue%d" % i for i in range(booster_count))
        )

    def translate(self, file):
        """Translate a gradient boosting model and write SAS scoring code to
        a file.

        Attributes
        ----------
        file : file object
            Open file for writing output SAS code.

        """
        for booster_id, tree in self._iter_trees():
            file.write("/* Parsing tree {}*/\n".format(booster_id))

            self._tree_parser.init(tree, booster_id)
            self._tree_parser.parse_node(file)

            file.write("\n")

        file.write("/* Getting target probability */\n")
        file.write(self._aggregate(booster_id + 1))
        file.write(
            "{} = {};\n".format(
                self.out_var_name, self.out_transform.format("treeValue")
            )
        )
