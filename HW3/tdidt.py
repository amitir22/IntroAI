from numpy import ndarray
from typing import Callable, List, Tuple, Union

from utilities import INVALID_FEATURE_INDEX, DEFAULT_MEAN_VALUE, SICK, HEALTHY, select_sick_examples, \
    are_equal_or_complement, classify_by_majority


class TDIDTree:
    """
    TDIDT - Top Down Induction Decision Tree
    """
    # binary-tree:
    left_subtree: Union['TDIDTree', None]
    right_subtree: Union['TDIDTree', None]

    # algorithm parameters:
    is_with_pruning: bool
    prune_threshold: int

    # node-data:
    num_examples: int
    num_sick_examples: int
    num_healthy_examples: int
    feature_index: int
    feature_split_value: float
    assigned_class: Union[HEALTHY, SICK]
    is_homogenous: bool

    # properties:

    @property
    def is_leaf(self):
        return self.left_subtree is None and self.right_subtree is None

    @property
    def is_not_leaf(self):
        return not self.is_leaf

    @property
    def leaf_count(self):  # mostly used for debugging
        if self.is_leaf:
            return 1

        left_leaf_count = 0
        right_leaf_count = 0

        if self.left_subtree is not None:
            left_leaf_count = self.left_subtree.leaf_count

        if self.right_subtree is not None:
            right_leaf_count = self.right_subtree.leaf_count

        return left_leaf_count + right_leaf_count + 1

    # methods:

    def __init__(self, is_with_pruning: bool, prune_threshold: int):
        """
        initializing algorithm parameters

        :param is_with_pruning: whether there's early pruning with regards to size or not
        :param prune_threshold: the critical number of examples to force pruning (ignored if is_with_pruning is False)
        """
        self.is_with_pruning = is_with_pruning
        self.prune_threshold = prune_threshold
        self.left_subtree = None
        self.right_subtree = None

    def generate_tree(self, examples: ndarray, features_indexes: List[int],
                      select_feature_func: Callable[[ndarray, List[int]], Tuple[int, float]],
                      default_classification: Union[SICK, HEALTHY]):
        """
        recursively builds the TDIDT for the given parameters

        :param examples: the given examples
        :param features_indexes: the indexes of the features we examine
        :param select_feature_func: the function used to split the examples to left/right
        :param default_classification: the default classification of the calling parent node
        """
        self.num_examples = len(examples)

        sick_examples = select_sick_examples(examples)

        self.num_sick_examples = len(sick_examples)
        self.num_healthy_examples = self.num_examples - self.num_sick_examples

        is_prune_needed = self.is_with_pruning and self.num_examples < self.prune_threshold

        self.is_homogenous = are_equal_or_complement(self.num_examples, self.num_sick_examples)

        # will be updated if needed be
        self.feature_index = INVALID_FEATURE_INDEX
        self.feature_split_value = DEFAULT_MEAN_VALUE

        if self.num_examples == 0 or is_prune_needed:
            self.assigned_class = default_classification
        elif self.is_homogenous:
            self.assigned_class = classify_by_majority(examples, sick_examples)
        else:
            self.assigned_class = classify_by_majority(examples, sick_examples)

            select_best_feature = select_feature_func
            did_exclude_feature = False

            self.feature_index, self.feature_split_value = select_best_feature(examples, features_indexes)

            left_examples = examples[examples[:, self.feature_index] < self.feature_split_value]
            right_examples = examples[examples[:, self.feature_index] >= self.feature_split_value]

            self.left_subtree = TDIDTree(is_with_pruning=self.is_with_pruning, prune_threshold=self.prune_threshold)
            self.right_subtree = TDIDTree(is_with_pruning=self.is_with_pruning, prune_threshold=self.prune_threshold)

            self.left_subtree.generate_tree(examples=left_examples, features_indexes=features_indexes,
                                            select_feature_func=select_feature_func,
                                            default_classification=self.assigned_class)
            self.right_subtree.generate_tree(examples=right_examples, features_indexes=features_indexes,
                                             select_feature_func=select_feature_func,
                                             default_classification=self.assigned_class)

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
        if self.is_not_leaf:
            example_feature_value = example[self.feature_index]

            if example_feature_value < self.feature_split_value and self.left_subtree is not None:
                return self.left_subtree.classify_single(example)
            if example_feature_value >= self.feature_split_value and self.right_subtree is not None:
                return self.right_subtree.classify_single(example)

        return self.assigned_class
