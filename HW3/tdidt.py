from numpy import ndarray
from typing import List, Union
from sklearn.model_selection import KFold
from feature_selector import FeatureSelector
from decision_tree_node import DecisionTreeNode
from utilities import INVALID_FEATURE_INDEX, DEFAULT_MEAN_VALUE, SICK, HEALTHY, classify_by_majority, \
                      select_sick_examples, is_homogenous


# TODO:
class TDIDTree:
    """
    TDIDT - Top Down Induction Decision Tree

    __init__ will recursively build a TDIDT tree using a given FeatureSelector
    """
    root_node: DecisionTreeNode

    # TODO: change 'feature_selector' to 'select_feature_func'
    def __init__(self, examples: ndarray, features_indexes: List[int], feature_selector: FeatureSelector,
                 default_classification: Union[SICK, HEALTHY]):
        """
        recursively builds a TDIDT
        """
        num_examples = len(examples)

        if num_examples == 0:
            self.root_node = DecisionTreeNode(num_examples, INVALID_FEATURE_INDEX, DEFAULT_MEAN_VALUE, None, None,
                                              default_classification, True)
        else:
            sick_examples = select_sick_examples(examples)
            num_sick_examples = len(sick_examples)
            default_classification = classify_by_majority(examples, sick_examples)

            if is_homogenous(num_examples, num_sick_examples):
                self.root_node = DecisionTreeNode(num_examples, INVALID_FEATURE_INDEX, DEFAULT_MEAN_VALUE, None, None,
                                                  default_classification, True)
            else:
                select_best_feature = feature_selector.select_best_feature_for_split

                best_feature_index, best_feature_mean_value = select_best_feature(examples, features_indexes)

                left_examples = examples[examples[:, best_feature_index] < best_feature_mean_value]
                right_examples = examples[examples[:, best_feature_index] >= best_feature_mean_value]

                left_subtree = TDIDTree(left_examples, features_indexes, feature_selector, default_classification)
                right_subtree = TDIDTree(right_examples, features_indexes, feature_selector, default_classification)

                left_node, right_node = left_subtree.root_node, right_subtree.root_node

                self.root_node = DecisionTreeNode(num_examples, best_feature_index, best_feature_mean_value, left_node,
                                                  right_node, default_classification, False)

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
        """
        classifying a singular given example using the decision tree

        :param example: the example to classify

        :return: a classifications (Union[utilities.SICK, utilities.HEALTHY])
        """
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
