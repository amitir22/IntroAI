from typing import List, Tuple
from numpy import ndarray, argsort
from abc import ABC

from utilities import INVALID_FEATURE_INDEX, DEFAULT_MEAN_VALUE, DEFAULT_INFO_GAIN
from info_gain_calculator import InfoGainCalculator


class FeatureSelector(ABC):
    """
    abstract feature selector class
    """
    def select_best_feature_for_split(self, examples: ndarray, features_indexes: List[int]) -> Tuple[int, float]:
        pass


class ID3FeatureSelector(FeatureSelector):
    """
    implementation of feature selector class for ID3
    """

    info_gain_calculator: InfoGainCalculator

    def __init__(self, info_gain_calculator: InfoGainCalculator):
        """
        initializing the info_gain_calculator of the feature selector with the given calculator

        :param info_gain_calculator: the given calculator
        """
        self.info_gain_calculator = info_gain_calculator

    def select_best_feature_for_split(self, examples: ndarray, features_indexes: List[int]) -> Tuple[int, float]:
        """
        selecting a feature to split by to get maximum info-gain

        :param examples: the examples we mean to split
        :param features_indexes: a list of the features indexes

        :return: two values: (1), (2) (Tuple[int, float])
                 (1): the index of the best feature to split by
                 (2): the best mean value to split by
        """
        best_feature_index = INVALID_FEATURE_INDEX
        best_info_gain = DEFAULT_INFO_GAIN
        best_feature_mean_value = DEFAULT_MEAN_VALUE

        for feature_index in features_indexes:
            current_feature_column = examples[:, feature_index]

            # bugfix: apparently trying to use 'sorted' corrupts the 'examples' table
            sorted_examples_indexes = argsort(current_feature_column)

            for i in range(len(sorted_examples_indexes) - 1):
                current_index = sorted_examples_indexes[i]
                next_index = sorted_examples_indexes[i + 1]

                current_feature_value = current_feature_column[current_index]
                next_feature_value = current_feature_column[next_index]

                mean_value = (current_feature_value + next_feature_value) / 2

                current_info_gain = self.info_gain_calculator.calc_info_gain(examples, feature_index, mean_value)

                if (current_info_gain, feature_index) == (best_info_gain, best_feature_index):
                    continue  # because we're told to choose the minimal feature index
                elif current_info_gain >= best_info_gain:
                    best_feature_mean_value = mean_value
                    best_info_gain = current_info_gain
                    best_feature_index = feature_index

        # todo: remove
        assert best_feature_index != INVALID_FEATURE_INDEX
        assert best_feature_mean_value != DEFAULT_MEAN_VALUE

        # todo: consider changing the order
        return best_feature_index, best_feature_mean_value
