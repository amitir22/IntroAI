from numpy import ndarray, round, array
from numpy.linalg import norm
from typing import Union
from copy import deepcopy


# global constants:
SICK = 'M'
HEALTHY = 'B'
STATUS_FEATURE_INDEX = 0
FIRST_NON_STATUS_FEATURE_INDEX = 1
INVALID_FEATURE_INDEX = -1
NO_INFO_GAIN = 0
DEFAULT_MEAN_VALUE = 0
FLOATING_POINT_ERROR_RANGE = 10 ** (-10)

# for ex3:
DEFAULT_WITHOUT_PRUNING = False
DEFAULT_PRUNE_THRESHOLD = -1
# M_VALUES_FOR_PRUNING = [2, 10, 30, 80, 160]  # consider using the half-series of 343 (170, 85, ...)
M_VALUES_FOR_PRUNING = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
DEFAULT_N_SPLIT = 5
DEFAULT_SHUFFLE = True
ID_SEED = 123456789  # todo: make sure to update to mine when needed:

# for ex4:
# todo: make sure to set the factor
FALSE_NEGATIVE_COST_FACTOR = 10  # used for calculating the loss function


# helper functions:

def select_sick_examples(examples: ndarray):
    """
    selecting the sick examples out of the given examples

    :param examples: the given examples to filter

    :return: only the sick examples (np.ndarray)
    """
    return examples[examples[:, STATUS_FEATURE_INDEX] == SICK]


def classify_by_majority(examples: ndarray, sick_examples: ndarray) -> Union[SICK, HEALTHY]:
    """
    this function determines the most common classification among the given examples by finding which classification
    has majority

    :param examples: the given examples
    :param sick_examples: the given sick examples selected from the given examples, can assume it's always true.

    :return: the classification determined by majority (str)
    """
    num_examples = len(examples)
    num_sick_examples = len(sick_examples)

    sick_ratio = num_sick_examples / num_examples

    if round(sick_ratio):
        return SICK  # if sick ratio >= 0.5
    else:
        return HEALTHY  # if sick ratio < 0.5


def are_equal_or_complement(num_group: int, num_subgroup: int):
    """
    checking whether the examples are homogenous.

    :param num_group: total number of examples
    :param num_subgroup: number of sick examples

    :return: True if homogenous, False otherwise. (bool)
    """
    return num_subgroup in [num_group, 0]  # if all/none of the examples are sick


def calc_examples_dist(example1: array, example2: array):
    """
    calculating the euclidean distance between the vector of example1 and example2 by calculating the norm of the
    difference vector

    :param example1: the vector of the first example
    :param example2: the vector of the second example

    :return: the euclidean distance (float)
    """
    diff = example1 - example2

    diff_norm = norm(diff)

    return diff_norm


def calc_centroid(examples: array):
    """
    calculating the centroid of the given examples

    :param examples: the given examples to create a centroid

    :return: the centroid vector (tuple)
    """
    num_examples = len(examples)

    sum_vector = sum(examples[:, STATUS_FEATURE_INDEX + 1:])

    centroid = sum_vector / num_examples

    return tuple(centroid)


def calc_error_rate(test_data: ndarray, test_results: list):
    """
    calculating the error-rate of the prediction test_results

    :param test_data: the original test data with the actual results
    :param test_results: the classifications predicted by the model

    :return: the error-rate (float)
    """
    total_count = len(test_data)

    # todo remove
    status_column = test_data[:, STATUS_FEATURE_INDEX].tolist()

    errors_indexes = get_errors_indexes(test_data, test_results)
    error_count = len(errors_indexes)

    error_rate = error_count / total_count

    return error_rate


def calc_loss(test_data: ndarray, test_results: list):
    """
    calculating the loss of the prediction test_results due to the cost function in ex4

    :param test_data: the original test data with the actual results
    :param test_results: the classifications predicted by the model

    :return: the loss (float)
    """
    total_count = len(test_data)

    errors_indexes = get_errors_indexes(test_data, test_results)

    false_positive_cost = 0
    false_negative_cost = 0

    for error_index in errors_indexes:
        if test_data[error_index, STATUS_FEATURE_INDEX] is SICK:
            # means we predicted a patient as healthy while he was sick (worse than false-positive)
            false_negative_cost += FALSE_NEGATIVE_COST_FACTOR
        else:
            # means we predicted a patient as sick while he was healthy
            false_positive_cost += 1

    total_cost = false_negative_cost + false_positive_cost

    loss = total_cost / total_count

    return loss


def get_errors_indexes(test_data: ndarray, test_results: list):
    """
    filtering the test results from the successful results, leaving only the errors, then returning a list of indexes
    of those results

    :param test_data: the original test data with the actual results
    :param test_results: the classifications predicted by the model

    :return: the indexes of the wrong predictions (list)
    """
    total_count = len(test_data)

    return [row_index for row_index in range(total_count)
            if test_results[row_index] != test_data[row_index, STATUS_FEATURE_INDEX]]


def is_within_floating_point_error_range(value: float):
    return -FLOATING_POINT_ERROR_RANGE <= value <= FLOATING_POINT_ERROR_RANGE


def utilities_test_zone():
    test_examples = array([[1, 2, 3], [4, 5, 6], [9, 8, 7]])
    cent = calc_centroid(test_examples)
    print(cent)


if __name__ == '__main__':
    utilities_test_zone()
