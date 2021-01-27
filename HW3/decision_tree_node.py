from typing import Union

from utilities import HEALTHY, SICK


# todo: check if dataclass is allowed
# todo: maybe add more data members to not be too similar
class DecisionTreeNode:
    """
    dataclass for each node in TDIDTree
    """
    num_examples: int
    num_sick_examples: int
    num_healthy_examples: int
    feature_index: int  # todo: sure? consider lambda
    feature_split_value: float
    left_sub_dt_tree: Union['DecisionTreeNode', None]
    right_sub_dt_tree: Union['DecisionTreeNode', None]
    assigned_class: Union[HEALTHY, SICK]
    is_homogenous: bool

    @property
    def is_leaf(self):
        return self.left_sub_dt_tree is None and self.right_sub_dt_tree is None

    @property
    def leaf_count(self):
        if self.is_leaf:
            return 1

        left_leaf_count = 0
        right_leaf_count = 0

        if self.left_sub_dt_tree is not None:
            left_leaf_count = self.left_sub_dt_tree.leaf_count

        if self.right_sub_dt_tree is not None:
            right_leaf_count = self.right_sub_dt_tree.leaf_count

        return left_leaf_count + right_leaf_count + 1

    def __init__(self, num_examples: int, num_sick_examples: int, num_healthy_examples: int, feature_index: int,
                 feature_split_value: float, left_sub_dt_tree: Union['DecisionTreeNode', None],
                 right_sub_dt_tree: Union['DecisionTreeNode', None], assigned_class: Union[HEALTHY, SICK],
                 is_homogenous: bool):
        self.num_examples = num_examples
        self.num_sick_examples = num_sick_examples
        self.num_healthy_examples = num_healthy_examples
        self.feature_index = feature_index
        self.feature_split_value = feature_split_value
        self.left_sub_dt_tree = left_sub_dt_tree
        self.right_sub_dt_tree = right_sub_dt_tree
        self.assigned_class = assigned_class
        self.is_homogenous = is_homogenous
