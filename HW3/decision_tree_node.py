from dataclasses import dataclass
from typing import Union
from utilities import HEALTHY, SICK


# todo: check if dataclass is allowed
# todo: maybe add more data members to not be too similar
@dataclass(init=True)
class DecisionTreeNode:
    """
    dataclass for each node in TDIDTree
    """
    num_examples: int
    feature_index: int  # todo: sure? consider lambda
    feature_split_value: float
    left_sub_dt_tree: Union['DecisionTreeNode', None]
    right_sub_dt_tree: Union['DecisionTreeNode', None]
    assigned_class: Union[HEALTHY, SICK]
    is_homogenous: bool
