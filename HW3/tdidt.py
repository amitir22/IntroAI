from numpy import ndarray
from typing import Callable, List, Tuple, Union

from decision_tree_node import DecisionTreeNode
from utilities import INVALID_FEATURE_INDEX, DEFAULT_MEAN_VALUE, SICK, HEALTHY, select_sick_examples, \
    are_equal_or_complement


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
                 default_classification_function: Callable[[ndarray, ndarray], Union[SICK, HEALTHY]],
                 excluded_feature_index: int):
        """
        recursively builds a TDIDT
        """
        num_examples = len(examples)

        is_prune_needed = is_with_pruning and num_examples < prune_threshold

        if num_examples == 0 or is_prune_needed:
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

            if are_equal_or_complement(num_examples, num_sick_examples):
                self.root_node = DecisionTreeNode(num_examples=num_examples, num_sick_examples=num_sick_examples,
                                                  num_healthy_examples=num_healthy_examples,
                                                  feature_index=INVALID_FEATURE_INDEX,
                                                  feature_split_value=DEFAULT_MEAN_VALUE,
                                                  left_sub_dt_tree=None, right_sub_dt_tree=None,
                                                  assigned_class=default_classification, is_homogenous=True)
            else:
                select_best_feature = select_feature_func
                did_exclude_feature = False

                # making sure the same feature isn't selected twice in a row
                if excluded_feature_index in features_indexes:
                    features_indexes.remove(excluded_feature_index)
                    did_exclude_feature = True

                best_feature_index, best_feature_mean_value = select_best_feature(examples, features_indexes)

                if did_exclude_feature:
                    features_indexes.append(excluded_feature_index)

                left_examples = examples[examples[:, best_feature_index] < best_feature_mean_value]
                right_examples = examples[examples[:, best_feature_index] >= best_feature_mean_value]

                left_subtree = TDIDTree(examples=left_examples, features_indexes=features_indexes,
                                        select_feature_func=select_feature_func,
                                        default_classification=default_classification,
                                        is_with_pruning=is_with_pruning, prune_threshold=prune_threshold,
                                        default_classification_function=default_classification_function,
                                        excluded_feature_index=best_feature_index)
                right_subtree = TDIDTree(examples=right_examples, features_indexes=features_indexes,
                                         select_feature_func=select_feature_func,
                                         default_classification=default_classification,
                                         is_with_pruning=is_with_pruning, prune_threshold=prune_threshold,
                                         default_classification_function=default_classification_function,
                                         excluded_feature_index=best_feature_index)

                left_node, right_node = left_subtree.root_node, right_subtree.root_node

                self.root_node = DecisionTreeNode(num_examples=num_examples, num_sick_examples=num_sick_examples,
                                                  num_healthy_examples=num_healthy_examples,
                                                  feature_index=best_feature_index,
                                                  feature_split_value=best_feature_mean_value,
                                                  left_sub_dt_tree=left_node, right_sub_dt_tree=right_node,
                                                  assigned_class=default_classification, is_homogenous=False)

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
