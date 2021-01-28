from pandas import DataFrame
from numpy import array, ndarray, argsort, linspace, round
from numpy.random import shuffle
from typing import Dict
from copy import deepcopy

from KNNForest import KNNForest
from tdidt import TDIDTree
from feature_selector import ID3FeatureSelector
from entropy_calculator import EntropyCalculator
from data_set_handler import DataSetHandler
from utilities import FIRST_NON_STATUS_FEATURE_INDEX, SICK, HEALTHY, WITHOUT_PRUNING, DEFAULT_PRUNE_THRESHOLD, \
    NUM_TESTS, K_FACTOR_RANGE, N_RANGE, p_RANGE, OPTIMAL_K, OPTIMAL_N, OPTIMAL_p, calc_centroid, calc_error_rate, \
    classify_by_sick_ratio


class ImprovedKNNForest(KNNForest):
    """
    an improved version of the KNNForest using weighted influence of each tree used for classifying
    """
    error_rates: Dict[tuple, float]

    def __init__(self, p_param: float, N_param: int, K_param: int, feature_selector: ID3FeatureSelector,
                 is_with_pruning: bool, prune_threshold: int):

        super().__init__(p_param, N_param, K_param, feature_selector, is_with_pruning, prune_threshold)

        self.error_rates = {}

    def train(self, dataset: DataFrame):
        """
        an improved version of the train method

        :param dataset: the dataset used to train the model
        """
        examples = dataset.to_numpy()

        first_column_index = FIRST_NON_STATUS_FEATURE_INDEX
        last_column_index = len(dataset.columns)

        features_indexes = list(range(first_column_index, last_column_index))

        num_examples = len(examples)  # n = num_examples

        # p * n = tree_population_size
        tree_population_size = int(round(self.tree_population_ratio_to_id3 * num_examples))

        for train_examples, test_examples in self.roll_random_train_and_test_examples(examples, tree_population_size):
            centroid_tuple = calc_centroid(train_examples)

            self.forest[centroid_tuple] = TDIDTree(is_with_pruning=self.is_with_pruning,
                                                   prune_threshold=self.prune_threshold)
            self.forest[centroid_tuple].generate_tree(examples=train_examples, features_indexes=features_indexes,
                                                      select_feature_func=self.select_feature_func,
                                                      default_classification=SICK)

            test_results = self.forest[centroid_tuple].classify(test_examples)

            error_rate = calc_error_rate(test_examples, test_results)

            self.error_rates[centroid_tuple] = error_rate

    def test_single(self, example: ndarray):
        """
        an improved version of the test_single method.
        here lies the main difference from the original implementation considering the influence of every tree
        in the forest in the classification process by weighing them with regards to prediction rates and distances.

        :param example: the given example to classify

        :return: the classification (Union[SICK, HEALTHY])
        """
        flat_example = example.flatten()

        distances, centroids_tuples = self.calc_distances_from_centroids(flat_example)

        max_distance = max(distances)
        min_distance = min(distances) * 0.99  # here to avoid dividing by 0

        sorted_tree_indexes = argsort(distances)[:self.num_trees_to_test]  # first k indexes of sorted by distance

        classifications_scores = {SICK: 0, HEALTHY: 0}

        for tree_index in sorted_tree_indexes:
            current_centroid_tuple = centroids_tuples[tree_index]
            current_distance = distances[tree_index]
            current_error_rate = self.error_rates[current_centroid_tuple]

            current_tree_result = self.forest[current_centroid_tuple].classify_single(example)

            current_tree_weight = (max_distance - min_distance) / (current_distance - min_distance)

            classifications_scores[current_tree_result] += (1 - current_error_rate) * current_tree_weight

        total_score = sum(classifications_scores.values())
        sick_score = classifications_scores[SICK]

        sick_ratio = sick_score / total_score

        return classify_by_sick_ratio(sick_ratio)

    # helper methods:
    def roll_random_train_and_test_examples(self, examples: array, num_random_examples: int):
        """

        :param examples: the given examples to select from
        :param num_random_examples: the number of examples to yield as the train_examples

        :return: a generator, yielding self.num_trees of random num_random_examples examples divided to:
                 train_examples, +validation_examples
        """
        examples_copy = deepcopy(examples)

        for i in range(self.num_trees):
            shuffle(examples_copy)

            train_examples = examples_copy[:num_random_examples, :]
            validation_examples = examples_copy[num_random_examples:, :]

            yield train_examples, validation_examples


def run_improved_knn_forest_with_params(data_handler: DataSetHandler, p_param: float, N_param: int, K_param: int):
    """
    training and testing the ImprovedKNNForest model

    :param data_handler: self explanatory
    :param p_param: self explanatory
    :param N_param: self explanatory
    :param K_param: self explanatory
    """
    K_param = min(K_param, N_param)

    # dependency injection
    info_gain_calculator = EntropyCalculator()
    id3_feature_selector = ID3FeatureSelector(info_gain_calculator)

    knn_forest = ImprovedKNNForest(p_param, N_param, K_param, id3_feature_selector, WITHOUT_PRUNING,
                                   DEFAULT_PRUNE_THRESHOLD)

    train_data, test_data = data_handler.read_both_data()

    knn_forest.train(train_data)

    test_results = knn_forest.test(test_data)

    error_rate = calc_error_rate(test_data.to_numpy(), test_results)
    prediction_rate = 1 - error_rate

    print(prediction_rate)


def run_experiments_on_improved_knn_forest(data_handler: DataSetHandler):
    """
    running experiments on the ImprovedKNNForest model parameters

    :param data_handler: self explanatory
    """
    p_min, p_max = p_RANGE
    p_range = linspace(p_min, p_max, NUM_TESTS)
    n_range = N_RANGE

    for p_param in p_range:
        for n_param in n_range:
            for k_factor in K_FACTOR_RANGE:
                k_param = int(round(n_param * k_factor))

                print(f'p={p_param}, N={n_param}, K={k_param}')
                run_improved_knn_forest_with_params(data_handler, p_param, n_param, k_param)


def ex7(data_handler: DataSetHandler):
    """
    the function that will run when running this file, running the model with the optimal parameters

    :param data_handler: self explanatory
    """
    run_improved_knn_forest_with_params(data_handler, OPTIMAL_p, OPTIMAL_N, OPTIMAL_K)


if __name__ == '__main__':
    data_set_handler = DataSetHandler()
    ex7(data_set_handler)
