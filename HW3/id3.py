import pandas
import utilities
from tdidt import TDIDTree
from feature_selector import FeatureSelector
from entropy_calculator import EntropyCalculator
from data_set_handler import DataSetHandler
from numpy import ndarray


class ID3:
    is_initialized: bool
    decision_tree: TDIDTree
    feature_selector: FeatureSelector

    def __init__(self, feature_selector: FeatureSelector):
        self.feature_selector = feature_selector

    def train(self, dataset: pandas.DataFrame):
        examples = dataset.to_numpy()

        first_column_index = utilities.STATUS_FEATURE_INDEX + 1
        last_column_index = len(dataset.columns)  # todo: maybe need to add +1? - no because start from 0

        features_indexes = list(range(first_column_index, last_column_index))

        self.decision_tree = TDIDTree(examples, features_indexes, self.feature_selector, utilities.SICK)
        self.is_initialized = True

    def test(self, dataset: pandas.DataFrame):
        examples = dataset.to_numpy()

        if self.is_initialized:
            return self.decision_tree.classify(examples)
        else:
            raise NotImplementedError('object of type ID3 is not initialized')


def ex1(data_handler: DataSetHandler):
    # dependency injection
    info_gain_calculator = EntropyCalculator()
    id3_feature_selector = FeatureSelector(info_gain_calculator)
    id3 = ID3(id3_feature_selector)

    train_data = data_handler.read_train_data()
    test_data = data_handler.read_test_data()

    id3.train(train_data)

    test_result = id3.test(test_data)

    print_ex1_test_result(test_data.to_numpy(), test_result)


def print_ex1_test_result(test_data: ndarray, test_result: list):
    error_count = 0
    total_count = len(test_data)

    for row_index in range(total_count):
        current_predicted = test_result[row_index]
        current_actual = test_data[row_index, utilities.STATUS_FEATURE_INDEX]

        if current_actual != current_predicted:
            error_count += 1

    error_rate = error_count / total_count
    prediction_rate = 1 - error_rate

    # todo: don't forget to eliminate all the text
    print(f'prediction rate: {prediction_rate}')


if __name__ == '__main__':
    data_set_handler = DataSetHandler()
    ex1(data_set_handler)
