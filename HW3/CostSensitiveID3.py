# todo: implement
from pandas import DataFrame
from random import randint, seed
from typing import List, Union

from ID3 import ID3
from feature_selector import ID3FeatureSelector
from entropy_calculator import EntropyCalculator
from utilities import ID_SEED, SICK, HEALTHY, DEFAULT_COST_MAJORITY_FACTOR, classify_by_cost_majority, calc_loss, \
    classify_by_majority
from data_set_handler import DataSetHandler


# TODO: test and document
class CostSensitiveID3(ID3):
    """
    ID3 Model of TDIDT with consideration to the loss function described in ex. 4
    """
    random_seed: int
    flip_mask: List[Union[SICK, HEALTHY]]

    def __init__(self, feature_selector: ID3FeatureSelector, random_seed: int):
        super().__init__(feature_selector)
        self.default_classification_function = classify_by_cost_majority
        self.random_seed = random_seed
        self.flip_mask = [SICK] * DEFAULT_COST_MAJORITY_FACTOR + [HEALTHY]  # consider making it a const

        seed(random_seed)

    def test(self, dataset: DataFrame):
        """
        randomly flipping the healthy results from the super-class.test() method to sick results to minimize the loss
        function described in ex. 4

        :param dataset: the data-set we use to test our model

        :return: the classifications
        """
        classifications = super(CostSensitiveID3, self).test(dataset)

        for c_index in range(len(classifications)):
            if classifications[c_index] is HEALTHY:
                classifications[c_index] = self.roll_result_flip()

        return classifications

    def roll_result_flip(self) -> Union[SICK, HEALTHY]:
        """
        rolling to flip a result classified as healthy to be classified as sick
        when the probability to flip from healthy to sick is 1 / (DEFAULT_COST_MAJORITY_FACTOR + 1)

        :return: the new classification (Union[SICK, HEALTHY])
        """
        random_index = randint(0, DEFAULT_COST_MAJORITY_FACTOR)

        return self.flip_mask[random_index]


# TODO:
def ex4(data_handler: DataSetHandler):
    # dependency injection
    info_gain_calculator = EntropyCalculator()
    id3_feature_selector = ID3FeatureSelector(info_gain_calculator)
    cs_id3 = CostSensitiveID3(id3_feature_selector, ID_SEED)
    id3 = ID3(id3_feature_selector)

    train_data, test_data = data_handler.read_both_data()

    cs_id3.train(train_data)
    id3.train(train_data)

    cs_tests_results = cs_id3.test(test_data)
    tests_results = id3.test(test_data)

    test_data_np = test_data.to_numpy()

    cs_loss = calc_loss(test_data_np, cs_tests_results)
    loss = calc_loss(test_data_np, tests_results)

    print(f'cs_loss: {cs_loss}')
    print(f'loss: {loss}')


if __name__ == '__main__':
    data_set_handler = DataSetHandler()
    ex4(data_set_handler)
