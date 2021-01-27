# todo: implement
from pandas import DataFrame
from random import randint, seed
from typing import List, Union
from numpy import ndarray

from ID3 import ID3
from feature_selector import ID3FeatureSelector
from cost_sensitive_entropy_calculator import CostSensitiveEntropyCalculator
from entropy_calculator import EntropyCalculator
from utilities import calc_loss
from data_set_handler import DataSetHandler


# TODO: implement, test and document
class CostSensitiveID3(ID3):
    """
    ID3 Model of TDIDT with consideration to the loss function described in ex.4
    """


# TODO:
def ex4(data_handler: DataSetHandler):
    # dependency injection
    cs_info_gain_calculator = CostSensitiveEntropyCalculator()
    info_gain_calculator = EntropyCalculator()

    cs_id3_feature_selector = ID3FeatureSelector(cs_info_gain_calculator)
    id3_feature_selector = ID3FeatureSelector(info_gain_calculator)

    # cs_id3 = CostSensitiveID3(cs_id3_feature_selector, ID_SEED)
    id3 = ID3(id3_feature_selector, False, 0)
    cs_id3 = ID3(cs_id3_feature_selector, False, 0)

    train_data, test_data = data_handler.read_both_data()

    id3.train(train_data)
    cs_id3.train(train_data)

    tests_results = id3.test(test_data)
    cs_tests_results = cs_id3.test(test_data)

    test_data_np = test_data.to_numpy()

    loss = calc_loss(test_data_np, tests_results)
    cs_loss = calc_loss(test_data_np, cs_tests_results)

    print(f'loss: {loss}')
    print(f'cs_loss: {cs_loss}')
    print(f'ratio: {cs_loss / loss} (lower is better)')


if __name__ == '__main__':
    data_set_handler = DataSetHandler()
    ex4(data_set_handler)
