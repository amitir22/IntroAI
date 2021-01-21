import numpy as np


# global constants:
SICK = 'M'
HEALTHY = 'B'
STATUS_FEATURE_INDEX = 0
INVALID_FEATURE_INDEX = -1
DEFAULT_INFO_GAIN = 0
DEFAULT_MEAN_VALUE = 0


# helper functions:


def select_sick_examples(examples: np.ndarray):
    """
    selecting the sick examples out of the given examples

    :param examples: the given examples to filter

    :return: only the sick examples (np.ndarray)
    """
    return examples[examples[:, STATUS_FEATURE_INDEX] == SICK]


def classify_by_majority(examples: np.ndarray, sick_examples: np.ndarray = None):
    """
    this function determines the most common classification among the given examples by finding which classification
    has majority

    :param examples: the given examples
    :param sick_examples: the given sick examples selected from the given examples, can assume it's always true.

    :return: the classification determined by majority (str)
    """
    num_examples = len(examples)

    if sick_examples is None:
        sick_examples = select_sick_examples(examples)

    num_sick_examples = len(sick_examples)

    sick_ratio = num_sick_examples / num_examples

    if np.round(sick_ratio):
        return SICK  # if sick ratio >= 0.5
    else:
        return HEALTHY  # if sick ratio < 0.5


def is_homogenous(num_examples: int, num_sick_examples: int):
    """
    checking whether the examples are homogenous.

    :param num_examples: total number of examples
    :param num_sick_examples: number of sick examples

    :return: True if homogenous, False otherwise. (bool)
    """
    return num_sick_examples in [num_examples, 0]  # if all/none of the examples are sick
