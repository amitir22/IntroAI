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
    feature_index: int  # todo: sure? consider lambda
    feature_split_value: float
    left_sub_dt_tree: 'DecisionTreeNode'
    right_sub_dt_tree: 'DecisionTreeNode'
    assigned_class: Union[HEALTHY, SICK]
    is_homogenous: bool
