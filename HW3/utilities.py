import numpy as np


# global constants:
SICK = 'M'
HEALTHY = 'B'
STATUS_COLUMN_INDEX = 0
INVALID_COLUMN_INDEX = -1
DEFAULT_INFO_GAIN = 0
DEFAULT_MEAN_VALUE = 0


# helper functions:
def select_sick_examples(examples: np.ndarray):
    return examples[examples[:, STATUS_COLUMN_INDEX] == SICK]


def classify_by_majority(examples: np.ndarray):
    num_examples = len(examples)
    num_sick_examples = len(select_sick_examples(examples))

    sick_ratio = num_sick_examples / num_examples

    if np.round(sick_ratio):
        return SICK  # if sick ratio >= 0.5
    else:
        return HEALTHY  # if sick ratio < 0.5
