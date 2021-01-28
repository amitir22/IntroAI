from typing import List, Dict, Callable, Tuple, Union
from pandas import DataFrame
from numpy import ndarray, array, round, argsort, asarray
from numpy.random import seed, shuffle
from copy import deepcopy

from learning_classifier_model import LearningClassifierModel
from tdidt import TDIDTree
from feature_selector import ID3FeatureSelector
from entropy_calculator import EntropyCalculator
from data_set_handler import DataSetHandler
from utilities import ID_SEED, FIRST_NON_STATUS_FEATURE_INDEX, SICK, HEALTHY, DEFAULT_PRUNE_THRESHOLD, \
    DEFAULT_WITHOUT_PRUNING, K, N, p, calc_centroid, calc_examples_dist, calc_error_rate, \
    classify_by_sick_ratio


# todo: and document
class KNNForest(LearningClassifierModel):
    forest: Dict[tuple, TDIDTree]  # tuple is the centroid of the ID3 tree
    tree_population_ratio_to_id3: float  # the p parameter of the algorithm
    num_trees: int  # the N parameter of the algorithm
    num_trees_to_test: int  # the K parameter of the algorithm

    # todo: leave new-lines? document?
    is_with_pruning: bool
    prune_threshold: int
    select_feature_func: Callable[[ndarray, List[int]], Tuple[int, float]]

    def __init__(self, p_param: float, N_param: int, K_param: int, feature_selector: ID3FeatureSelector,
                 is_with_pruning: bool, prune_threshold: int):
        self.tree_population_ratio_to_id3 = p_param
        self.num_trees = N_param
        self.num_trees_to_test = K_param

        self.is_with_pruning = is_with_pruning
        self.prune_threshold = prune_threshold
        self.select_feature_func = feature_selector.select_best_feature_for_split
        self.forest = {}

        # TODO: remove?
        assert 0.3 <= p_param <= 0.7

        seed(ID_SEED)  # here to get consistent results

    def train(self, dataset: DataFrame):
        examples = dataset.to_numpy()

        first_column_index = FIRST_NON_STATUS_FEATURE_INDEX
        last_column_index = len(dataset.columns)

        features_indexes = list(range(first_column_index, last_column_index))

        num_examples = len(examples)  # n = num_examples

        # p * n = tree_population_size
        tree_population_size = int(round(self.tree_population_ratio_to_id3 * num_examples))

        for random_examples in self.get_random_examples(examples, tree_population_size):
            centroid_tuple = calc_centroid(random_examples)

            self.forest[centroid_tuple] = TDIDTree(is_with_pruning=self.is_with_pruning,
                                                   prune_threshold=self.prune_threshold)
            self.forest[centroid_tuple].generate_tree(examples=random_examples, features_indexes=features_indexes,
                                                      select_feature_func=self.select_feature_func,
                                                      default_classification=SICK)

    def test(self, dataset: DataFrame):
        examples = dataset.to_numpy()
        results = []

        for example in examples:
            example_result = self.test_single(example)

            results.append(example_result)

        return results

    # helper methods:
    def test_single(self, example: ndarray):
        flat_example = example.flatten()

        distances, centroids_tuples = self.calc_distances_from_centroids(flat_example)

        sorted_tree_indexes = argsort(distances)[:self.num_trees_to_test]  # first k indexes of sorted by distance

        tree_results = self.get_tree_results(sorted_tree_indexes, centroids_tuples, flat_example)

        tree_sick_ratio = self.calc_tree_sick_ratio(tree_results)

        return classify_by_sick_ratio(tree_sick_ratio)

    def calc_distances_from_centroids(self, flat_example: array):
        distances = []
        centroids_tuples = []

        for tree_centroid_tuple in self.forest:
            tree_centroid = asarray(tree_centroid_tuple)

            flat_example_without_status_column = flat_example[FIRST_NON_STATUS_FEATURE_INDEX:]

            euclidean_dist_centroid = calc_examples_dist(flat_example_without_status_column, tree_centroid)

            distances.append(euclidean_dist_centroid)
            centroids_tuples.append(tree_centroid_tuple)

        return distances, centroids_tuples

    def get_tree_results(self, sorted_tree_indexes: array, centroids: List[array], example: ndarray):
        tree_results = []

        for tree_index in sorted_tree_indexes:
            current_centroid_tuple = centroids[tree_index]

            tree_result = self.forest[current_centroid_tuple].classify_single(example)

            tree_results.append(tree_result)

        return tree_results

    def get_random_examples(self, examples: array, num_random_examples: int):
        examples_copy = deepcopy(examples)

        for i in range(self.num_trees):
            shuffle(examples_copy)

            sliced = examples_copy[:num_random_examples, :]

            yield sliced

    @staticmethod
    def calc_tree_sick_ratio(tree_results: List[Union[SICK, HEALTHY]]):
        num_tree_results = len(tree_results)
        num_sick_tree_results = len([tree_result for tree_result in tree_results
                                     if tree_result is SICK])

        tree_sick_ratio = num_sick_tree_results / num_tree_results

        return tree_sick_ratio


def run_knn_forest_with_parameters(data_handler: DataSetHandler, p_param: float, N_param: int, K_param: int):
    # dependency injection
    info_gain_calculator = EntropyCalculator()
    id3_feature_selector = ID3FeatureSelector(info_gain_calculator)

    knn_forest = KNNForest(p_param, N_param, K_param, id3_feature_selector, DEFAULT_WITHOUT_PRUNING,
                           DEFAULT_PRUNE_THRESHOLD)

    train_data, test_data = data_handler.read_both_data()

    knn_forest.train(train_data)

    test_results = knn_forest.test(test_data)

    error_rate = calc_error_rate(test_data.to_numpy(), test_results)
    prediction_rate = 1 - error_rate

    # todo: don't forget to eliminate all the text
    # print(f'prediction rate: {prediction_rate}')
    print(prediction_rate)


def run_experiments_on_knn_forest():
    pass


def ex6(data_handler: DataSetHandler):
    run_knn_forest_with_parameters(data_handler, p, N, K)


if __name__ == '__main__':
    data_set_handler = DataSetHandler()
    ex6(data_set_handler)
