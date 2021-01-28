from ID3 import ID3
from feature_selector import ID3FeatureSelector
from cost_sensitive_entropy_calculator import CostSensitiveEntropyCalculator
from utilities import DEFAULT_PRUNE_THRESHOLD, WITHOUT_PRUNING, calc_loss
from data_set_handler import DataSetHandler


class CostSensitiveID3(ID3):
    """
    ID3 Model of TDIDT with consideration to the loss function described in ex.4
    """

    def __init__(self, is_with_pruning: bool, prune_threshold: int):
        super().__init__(is_with_pruning, prune_threshold)

        info_gain_calculator = CostSensitiveEntropyCalculator()  # where the difference between the regular ID3 lies.
        id3_feature_selector = ID3FeatureSelector(info_gain_calculator)

        self.select_feature_func = id3_feature_selector.select_best_feature_for_split


def ex4(data_handler: DataSetHandler):
    """
    the function that will run when running this file

    :param data_handler: self explanatory
    """
    cs_id3 = CostSensitiveID3(WITHOUT_PRUNING, DEFAULT_PRUNE_THRESHOLD)

    train_data, test_data = data_handler.read_both_data()

    cs_id3.train(train_data)

    cs_tests_results = cs_id3.test(test_data)

    test_data_np = test_data.to_numpy()

    cs_loss = calc_loss(test_data_np, cs_tests_results)

    print(cs_loss)


if __name__ == '__main__':
    data_set_handler = DataSetHandler()
    ex4(data_set_handler)
