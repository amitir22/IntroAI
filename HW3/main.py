# List of todos for entire HW3
# TODO: eliminate all todos
# todo: make sure to document all important functions
# todo: ...

from id3 import ID3
from feature_selector import FeatureSelector
from entropy_calculator import EntropyCalculator
from data_set_handler import DataSetHandler
from numpy import ndarray
from utilities import SICK, HEALTHY


def ex1(data_set_handler: DataSetHandler):
    # dependency injection
    info_gain_calculator = EntropyCalculator()
    id3_feature_selector = FeatureSelector(info_gain_calculator)
    id3 = ID3(id3_feature_selector)

    id3.train(data_set_handler.read_train_data())
    test_result = id3.test(data_set_handler.read_test_data())

    assert_ex1_test_result(test_result)


def assert_ex1_test_result(test_result: ndarray):
    # todo: change
    print(test_result)


def main():
    data_set_handler = DataSetHandler()
    ex1(data_set_handler)


if __name__ == '__main__':
    main()
