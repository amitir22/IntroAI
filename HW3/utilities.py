from numpy import ndarray, round, array
from numpy.linalg import norm
from typing import Union


# global constants:
SICK = 'M'
HEALTHY = 'B'
STATUS_FEATURE_INDEX = 0
INVALID_FEATURE_INDEX = -1
DEFAULT_INFO_GAIN = 0
DEFAULT_MEAN_VALUE = 0

# for ex3:
DEFAULT_PRUNE_THRESHOLD = -1
M_VALUES_FOR_PRUNING = [2, 10, 30, 80, 160]  # consider using the half-series of 343 (170, 85, ...)
DEFAULT_N_SPLIT = len(M_VALUES_FOR_PRUNING)
DEFAULT_SHUFFLE = True
ID_SEED = 123456789  # todo: make sure to update to mine when needed:

# for ex4:
# todo: make sure to set the factor
FALSE_POSITIVE_COST_FACTOR = 5  # used for calculating the loss function
DEFAULT_COST_MAJORITY_FACTOR = FALSE_POSITIVE_COST_FACTOR  # used for adapting the ID3 to minimize loss function


# helper functions:

def select_sick_examples(examples: ndarray):
    """
    selecting the sick examples out of the given examples

    :param examples: the given examples to filter

    :return: only the sick examples (np.ndarray)
    """
    return examples[examples[:, STATUS_FEATURE_INDEX] == SICK]


def classify_by_majority(examples: ndarray, sick_examples: ndarray = None) -> Union[SICK, HEALTHY]:
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

    if round(sick_ratio):
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


def calc_examples_dist(example1: ndarray, example2: ndarray):
    """
    calculating the euclidean distance between the vector of example1 and example2 by calculating the norm of the
    difference vector

    :param example1: the vector as the matrix of the first example
    :param example2: the vector as the matrix of the second example

    :return: the euclidean distance (float)
    """
    ex1 = example1.flatten()
    ex2 = example2.flatten()

    diff = ex1 - ex2

    diff_norm = norm(diff)

    return diff_norm


# todo: document and implement
def calc_centroid(examples: ndarray):
    """
    calculating the centroid of the given examples

    :param examples: the given examples to create a centroid

    :return: the centroid vector (ndarray)
    """
    num_examples = len(examples)
    sum_vector = sum(examples)

    centroid = sum_vector / num_examples

    return centroid


def calc_error_rate(test_data: ndarray, test_results: list):
    """
    calculating the error-rate of the prediction test_results

    :param test_data: the original test data with the actual results
    :param test_results: the classifications predicted by the model

    :return: the error-rate (float)
    """
    total_count = len(test_data)

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
            # means we predicted a patient as healthy while he was sick
            false_negative_cost += 1
        else:
            # means we predicted a patient as sick while he was healthy
            false_positive_cost += (1 / FALSE_POSITIVE_COST_FACTOR)

    total_cost = false_negative_cost + false_positive_cost

    loss = total_cost / total_count

    return loss


def get_errors_indexes(test_data, test_results):
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


def classify_by_cost_majority(examples: ndarray, sick_examples: ndarray = None) -> Union[SICK, HEALTHY]:
    """
    this function determines the most common classification among the given examples by finding which classification
    has majority

    difference: except for determining by majority, it's determined by the cost majority where sick examples
                affect more than healthy examples

    :param examples: the given examples
    :param sick_examples: the given sick examples selected from the given examples, can assume it's always true.

    :return: the classification determined by majority (Union[SICK, HEALTHY])
    """
    num_examples = len(examples)

    if sick_examples is None:
        sick_examples = select_sick_examples(examples)

    num_sick_examples = len(sick_examples)
    num_healthy_examples = num_examples - num_sick_examples

    healthy_cost = num_healthy_examples
    sick_cost = num_sick_examples * DEFAULT_COST_MAJORITY_FACTOR

    total_cost = healthy_cost + sick_cost

    sick_ratio_with_cost = sick_cost / total_cost

    if round(sick_ratio_with_cost):
        return SICK  # if sick ratio >= 0.5
    else:
        return HEALTHY  # if sick ratio < 0.5


def utilities_test_zone():
    test_examples = array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    cent = calc_centroid(test_examples)
    print(cent)


if __name__ == '__main__':
    utilities_test_zone()
