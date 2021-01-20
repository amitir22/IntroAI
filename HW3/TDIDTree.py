import utilities
from numpy import ndarray
from typing import List
from feature_selector import FeatureSelector
from decision_tree_node import DecisionTreeNode


# TODO:
class TDIDTree:
    """
    TDIDT - Top Down Induction Decision Tree
    __init__ will recursively build a TDIDT tree using a given FeatureSelector
    """
    feature_selector: FeatureSelector
    root_node: DecisionTreeNode

    # TODO:
    def __init__(self, examples: ndarray, features_indexes: List[int], feature_selector: FeatureSelector):
        """
        recursively builds a TDIDT
        """
        self.feature_selector = feature_selector

    # TODO: check
    def classify(self, examples: ndarray):
        """
        classifying the given examples using the decision tree
        :param examples: the examples to classify
        :return: a list of the classifications (list containing utilities.SICK or utilities.HEALTHY for every example)
        """
        classifies = []

        for example in examples:
            classifies.append(self.classify_single(example))

        return classifies

    # helper functions:
    def classify_single(self, example: ndarray):
        current_node = self.root_node

        # todo: make sure i can use 'is' like this
        while current_node.is_homogenous is False:
            current_feature_index = current_node.feature_index
            current_feature_value = example[current_feature_index]

            # todo: make sure it's supposed to be '<' and not '<='
            if current_feature_value < current_node.feature_split_value:
                current_node = current_node.left_sub_dt_tree
            else:
                current_node = current_node.right_sub_dt_tree

        return current_node.assigned_class
