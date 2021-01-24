from numpy import ndarray
from typing import Callable, List, Tuple, Union

from decision_tree_node import DecisionTreeNode
from utilities import INVALID_FEATURE_INDEX, DEFAULT_MEAN_VALUE, SICK, HEALTHY, DEFAULT_PRUNE_THRESHOLD, \
                      select_sick_examples, is_homogenous


# TODO:
class TDIDTree:
    """
    TDIDT - Top Down Induction Decision Tree

    __init__ will recursively build a TDIDT tree using a given FeatureSelector
    """
    root_node: DecisionTreeNode

    def __init__(self, examples: ndarray, features_indexes: List[int],
                 select_feature_func: Callable[[ndarray, List[int]], Tuple[int, float]],
                 default_classification: Union[SICK, HEALTHY],
                 is_with_pruning: bool, prune_threshold: int,
                 default_classification_function: Callable[[ndarray, ndarray], Union[SICK, HEALTHY]]):
        """
        recursively builds a TDIDT
        """
        num_examples = len(examples)

        is_prune_needed = is_with_pruning and num_examples < prune_threshold

        if num_examples == 0 or is_prune_needed:
            assert prune_threshold > DEFAULT_PRUNE_THRESHOLD  # todo: remove

            self.root_node = DecisionTreeNode(num_examples=num_examples, num_sick_examples=num_examples,
                                              num_healthy_examples=num_examples, feature_index=INVALID_FEATURE_INDEX,
                                              feature_split_value=DEFAULT_MEAN_VALUE,
                                              left_sub_dt_tree=None, right_sub_dt_tree=None,
                                              assigned_class=default_classification, is_homogenous=True)
        else:
            sick_examples = select_sick_examples(examples)
            num_sick_examples = len(sick_examples)
            num_healthy_examples = num_examples - num_sick_examples
            default_classification = default_classification_function(examples, sick_examples)

            if is_homogenous(num_examples, num_sick_examples):
                self.root_node = DecisionTreeNode(num_examples=num_examples, num_sick_examples=num_sick_examples,
                                                  num_healthy_examples=num_healthy_examples,
                                                  feature_index=INVALID_FEATURE_INDEX,
                                                  feature_split_value=DEFAULT_MEAN_VALUE,
                                                  left_sub_dt_tree=None, right_sub_dt_tree=None,
                                                  assigned_class=default_classification, is_homogenous=True)
            else:
                select_best_feature = select_feature_func

                best_feature_index, best_feature_mean_value = select_best_feature(examples, features_indexes)

                left_examples = examples[examples[:, best_feature_index] < best_feature_mean_value]
                right_examples = examples[examples[:, best_feature_index] >= best_feature_mean_value]

                left_subtree = TDIDTree(left_examples, features_indexes, select_feature_func, default_classification,
                                        is_with_pruning, prune_threshold, default_classification_function)
                right_subtree = TDIDTree(right_examples, features_indexes, select_feature_func, default_classification,
                                         is_with_pruning, prune_threshold, default_classification_function)

                left_node, right_node = left_subtree.root_node, right_subtree.root_node

                self.root_node = DecisionTreeNode(num_examples=num_examples, num_sick_examples=num_sick_examples,
                                                  num_healthy_examples=num_healthy_examples,
                                                  feature_index=best_feature_index,
                                                  feature_split_value=best_feature_mean_value,
                                                  left_sub_dt_tree=left_node, right_sub_dt_tree=right_node,
                                                  assigned_class=default_classification, is_homogenous=True)

    def classify(self, examples: ndarray):
        """
        classifying the given examples using the decision tree

        :param examples: the examples to classify

        :return: a list of the classifications (list containing utilities.SICK or utilities.HEALTHY for every example)
        """
        classifications = []

        for example in examples:
            classifications.append(self.classify_single(example))

        return classifications

    # helper functions:

    def classify_single(self, example: ndarray):
        """
        classifying a singular given example using the decision tree

        :param example: the example to classify

        :return: a classifications (Union[utilities.SICK, utilities.HEALTHY])
        """
        current_node = self.root_node

        while current_node.is_homogenous is False:
            current_feature_index = current_node.feature_index
            current_feature_value = example[current_feature_index]

            if current_feature_value < current_node.feature_split_value:
                current_node = current_node.left_sub_dt_tree
            else:
                current_node = current_node.right_sub_dt_tree

        return current_node.assigned_class
