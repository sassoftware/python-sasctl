try:
    from lxml import objectify
except ImportError:
    objectify = None

from .core import TreeParser


class PmmlParser(TreeParser):
    """Class for parsing pmml gradient boosting models.

    Parameters
    ----------
    tree_root : etree.Element
        Root node of pmml gradient boosting forest.
    """
    def __init__(self, tree_root):
        super(PmmlParser, self).__init__()

        self._tree_root = tree_root
        for elem in tree_root.getiterator():
            if not hasattr(elem.tag, 'find'): continue
            i = elem.tag.find('}')
            if i >= 0:
                elem.tag = elem.tag[i+1:]
        if objectify is None:
            raise RuntimeError("Lxml is not installed. Lxml is needed to parse xml files.") 
        objectify.deannotate(tree_root, cleanup_namespaces=True)
        
        self._forest = tree_root.find('MiningModel/Segmentation')[0].find('MiningModel')
        
        rescaleConstant = self._forest.find('Targets/Target').get('rescaleConstant')
        self.out_transform = "1 / (1 + exp(-{}))".format("({0} + " + "{})".format(rescaleConstant))


    def _not_leaf(self, node):
        return node.get('defaultChild')


    def _get_var(self, node):
        return node.find('Node').find('SimplePredicate').get('field')


    def _go_left(self, node):
        return (node.find('Node').get('id') == node.get('defaultChild'))

    
    def _go_right(self, node):
        return (not node.find('Node').get('id') == node.get('defaultChild'))


    def _left_node(self, node):
        return node.findall('Node')[0]

    
    def _right_node(self, node):
        return node.findall('Node')[1]


    def _missing_node(self, node):
        return None

    
    def _split_value(self, node):
        return node.find('Node').find('SimplePredicate').get('value')


    def _decision_type(self, node):
        ops = {'lessThan': '<', 'lessOrEqual': '<=', 'greaterThan': '>', 'greaterOrEqual': '>='}
        return ops[node.find('Node').find('SimplePredicate').get('operator')]


    def _leaf_value(self, node):
        return node.get('score')

    
    def _iter_trees(self):
        for booster_id, tree_elem in enumerate(self._forest.find('Segmentation')):
            yield booster_id, tree_elem.find('TreeModel/Node')
