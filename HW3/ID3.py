from matplotlib import pyplot
from pandas import DataFrame
from numpy import ndarray
from typing import Callable, List, Tuple
from sklearn.model_selection import KFold

from learning_classifier_model import LearningClassifierModel
from tdidt import TDIDTree
from feature_selector import ID3FeatureSelector
from entropy_calculator import EntropyCalculator
from data_set_handler import DataSetHandler
from utilities import SICK, FIRST_NON_STATUS_FEATURE_INDEX, DEFAULT_N_SPLIT, DEFAULT_SHUFFLE, ID_SEED, \
    M_VALUES_FOR_PRUNING, DEFAULT_PRUNE_THRESHOLD, WITHOUT_PRUNING, calc_error_rate


class ID3(LearningClassifierModel):
    """
    ID3 Model of TDIDT
    """
    is_trained: bool
    select_feature_func: Callable[[ndarray, List[int]], Tuple[int, float]]
    decision_tree: TDIDTree

    def __init__(self, is_with_pruning: bool, prune_threshold: int):
        """
        initializing the model's parameters

        :param is_with_pruning: whether or not the model will execute early pruning
        :param prune_threshold: non-homogenous leaves will have less examples than the given value (prune_threshold)
        """
        info_gain_calculator = EntropyCalculator()
        id3_feature_selector = ID3FeatureSelector(info_gain_calculator)

        self.select_feature_func = id3_feature_selector.select_best_feature_for_split
        self.decision_tree = TDIDTree(is_with_pruning=is_with_pruning, prune_threshold=prune_threshold)

    def train(self, dataset: DataFrame):
        """
        training the model with the given dataset

        :param dataset: the given dataset
        """
        examples = dataset.to_numpy()

        first_column_index = FIRST_NON_STATUS_FEATURE_INDEX
        last_column_index = len(dataset.columns)

        features_indexes = list(range(first_column_index, last_column_index))

        self.decision_tree.generate_tree(examples=examples, features_indexes=features_indexes,
                                         select_feature_func=self.select_feature_func,
                                         default_classification=SICK)

        self.is_trained = True

    def test(self, dataset: DataFrame):
        """
        testing the model's classifications with the given dataset

        :param dataset: the given data set

        :return: the classifications of the examples in the dataset (List[Union[SICK, HEALTHY]])
        """
        examples = dataset.to_numpy()

        if self.is_trained:
            return self.decision_tree.classify(examples)
        else:
            raise NotImplementedError('forgot to train the model: ID3. use method train()')


def ex1(data_handler: DataSetHandler):
    """
    the function that will run when running this file

    :param data_handler: self explanatory
    """
    id3 = ID3(WITHOUT_PRUNING, DEFAULT_PRUNE_THRESHOLD)

    train_data, test_data = data_handler.read_both_data()

    id3.train(train_data)

    test_results = id3.test(test_data)

    numpy_test_data = test_data.to_numpy()

    error_rate = calc_error_rate(numpy_test_data, test_results)
    prediction_rate = 1 - error_rate

    print(prediction_rate)


def ex3(data_handler: DataSetHandler):
    """
    instructions: create an object of type DataSetHandler with the path to the test data and the train data and pass
                  him as argument

    :param data_handler: a DataSetHandler object handling the data-sets reading

    :return: None (plotting a graph without returning a value)
    """
    k_fold = KFold(DEFAULT_N_SPLIT, DEFAULT_SHUFFLE, ID_SEED)
    m_values = M_VALUES_FOR_PRUNING
    m_prediction_rates = []

    dataset = data_handler.read_train_data()
    examples = dataset.to_numpy()
    columns = dataset.columns  # for reconstructing a data-frame

    for m in m_values:
        prune_threshold = m
        sum_rates = 0

        for train_indexes, test_indexes in k_fold.split(examples):
            train_examples, test_examples = examples[train_indexes], examples[test_indexes]

            train_data = DataFrame(data=train_examples, columns=columns)
            test_data = DataFrame(data=test_examples, columns=columns)

            test_results = run_id3_with_pruning(train_data, test_data, prune_threshold)

            error_rate = calc_error_rate(test_examples, test_results)

            prediction_rate = 1 - error_rate

            sum_rates += prediction_rate

        average_rate = sum_rates / DEFAULT_N_SPLIT
        m_prediction_rates.append(average_rate)

    plot_m_results(m_values, m_prediction_rates)


def plot_m_results(m_values: List[int], m_prediction_rates: List[float]):
    pyplot.xlabel('M value')
    pyplot.ylabel('Prediction Rate')
    pyplot.plot(m_values, m_prediction_rates)
    pyplot.show()


def run_id3_with_pruning(train_data: DataFrame, test_data: DataFrame, prune_threshold: int):
    with_pruning = True
    id3 = ID3(with_pruning, prune_threshold)

    id3.train(train_data)

    test_results = id3.test(test_data)

    return test_results


if __name__ == '__main__':
    data_set_handler = DataSetHandler()
    ex1(data_set_handler)
