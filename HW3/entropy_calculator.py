from numpy import ndarray, log2
from typing import Tuple

from info_gain_calculator import InfoGainCalculator
from utilities import DEFAULT_INFO_GAIN, are_equal_or_complement, select_sick_examples


class EntropyCalculator(InfoGainCalculator):
    """
    an InfoGainCalculator implemented using entropy
    """

    def calc_info_gain(self, examples: ndarray, feature_index: int, mean_feature_value):
        """
        calculating the possible information gain from splitting the given examples by the given mean_feature_value

        :param examples: the table we analyze
        :param feature_index: the index of the feature (column) we use to calculate the information gained
        :param mean_feature_value: the value we split by to left and right

        :return: the "score" of the information gained (float)
        """
        num_examples = len(examples)

        left_examples = examples[examples[:, feature_index] < mean_feature_value]
        num_left_examples = len(left_examples)

        if are_equal_or_complement(num_examples, num_left_examples):  # here to save some calculation time
            return DEFAULT_INFO_GAIN

        right_examples = examples[examples[:, feature_index] >= mean_feature_value]
        num_right_examples = len(right_examples)

        num_sick, num_left_sick, num_right_sick = self.get_num_sick_examples(examples, left_examples, right_examples)

        # inferring the healthy parts sizes
        num_healthy = num_examples - num_sick
        num_left_healthy = num_left_examples - num_left_sick
        num_right_healthy = num_right_examples - num_right_sick

        # calculating costs
        current_total_cost, current_sick_cost, current_healthy_cost = self.calc_costs(num_sick, num_healthy)
        left_total_cost, left_sick_cost, left_healthy_cost = self.calc_costs(num_left_sick, num_left_healthy)
        right_total_cost, right_sick_cost, right_healthy_cost = self.calc_costs(num_right_sick, num_right_healthy)

        # calculation probabilities
        current_probabilities = self.calc_probabilities(current_sick_cost, current_healthy_cost)
        left_probabilities = self.calc_probabilities(left_sick_cost, left_healthy_cost)
        right_probabilities = self.calc_probabilities(right_sick_cost, right_healthy_cost)

        # calculating entropy
        current_entropy = self.calc_entropy(current_probabilities)
        left_entropy = self.calc_entropy(left_probabilities)
        right_entropy = self.calc_entropy(right_probabilities)

        left_ratio, right_ratio = self.calc_ratios(left_total_cost, right_total_cost)

        # the heart of this calculator:
        info_gain = current_entropy - (left_ratio * left_entropy + right_ratio * right_entropy)

        return info_gain

    # helper functions:

    @staticmethod
    def calc_entropy(probabilities: Tuple[float, float]):
        """
        calculating the entropy of the given probabilities
        usual usage is probabilities=[probability_sick, probability_healthy]

        :param probabilities: the probabilities

        :return: the entropy of the given probabilities (float)
        """
        cumulated_entropy = 0

        assert sum(probabilities) == 1

        for probability in probabilities:
            if probability != 0:
                cumulated_entropy -= probability * log2(probability)

        return cumulated_entropy

    # todo: document - important
    @staticmethod
    def get_num_sick_examples(examples: ndarray, left_examples: ndarray, right_examples: ndarray):
        sick_examples = select_sick_examples(examples)
        left_sick_examples = select_sick_examples(left_examples)
        right_sick_examples = select_sick_examples(right_examples)

        num_sick = len(sick_examples)
        num_left_sick = len(left_sick_examples)
        num_right_sick = len(right_sick_examples)

        return num_sick, num_left_sick, num_right_sick

    # todo: document - important
    @staticmethod
    def calc_probabilities(sick_cost: int, healthy_cost: int):
        total_cost = sick_cost + healthy_cost

        sick_probability = sick_cost / total_cost
        healthy_probability = healthy_cost / total_cost

        assert sick_probability + healthy_probability == 1

        return sick_probability, healthy_probability

    # todo: document - important
    @staticmethod
    def calc_ratios(num_left_examples: int, num_right_examples: int):
        num_examples = num_left_examples + num_right_examples
        
        left_ratio = num_left_examples / num_examples
        right_ratio = num_right_examples / num_examples

        return left_ratio, right_ratio

    # todo: document
    @staticmethod
    def calc_costs(num_sick_examples: int, num_healthy_examples: int):
        """
        a simple implementation mainly used to be overridden in ex.4
        assuming each example's cost is 1

        :param num_sick_examples: the number of sick examples
        :param num_healthy_examples: the number of healthy examples

        :return: tuple((1), (2), (3)):
                 (1): the total cost
                 (2): the sick cost
                 (3): the healthy cost
        """
        total_cost = num_sick_examples + num_healthy_examples

        return total_cost, num_sick_examples, num_healthy_examples
