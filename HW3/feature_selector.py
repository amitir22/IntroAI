import utilities
from info_gain_calculator import InfoGainCalculator
from numpy import ndarray, argsort


# TODO: maybe rename to ID3FeatureSelector
class FeatureSelector:
    """
    ID3 feature selector class
    """

    info_gain_calculator: InfoGainCalculator

    def __init__(self, entropy_calculator: InfoGainCalculator):
        self.info_gain_calculator = entropy_calculator

    def select_feature_for_split(self, examples: ndarray, features_indexes):
        best_feature_index = utilities.INVALID_COLUMN_INDEX
        best_info_gain = utilities.DEFAULT_INFO_GAIN
        best_feature_mean_value = utilities.DEFAULT_MEAN_VALUE

        for feature_index in features_indexes:
            current_feature_column = examples[:, feature_index]

            # apparently trying to use 'sorted' corrupts the 'examples' table
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
        assert best_feature_index != utilities.INVALID_COLUMN_INDEX
        assert best_feature_mean_value != utilities.DEFAULT_MEAN_VALUE

        # todo: consider changing the order
        return best_feature_index, best_feature_mean_value
