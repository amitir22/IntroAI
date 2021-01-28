from numpy import ndarray, log2
from typing import Tuple

from info_gain_calculator import InfoGainCalculator
from utilities import NO_INFO_GAIN, are_equal_or_complement, select_sick_examples, \
    is_within_floating_point_error_range


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
            return NO_INFO_GAIN

        right_examples = examples[examples[:, feature_index] >= mean_feature_value]
        num_right_examples = len(right_examples)

        # calculating entropy
        total_cost, current_entropy = self.calc_cost_entropy(examples)
        left_cost, left_entropy = self.calc_cost_entropy(left_examples)
        right_cost, right_entropy = self.calc_cost_entropy(right_examples)

        left_ratio, right_ratio = self.calc_ratios(left_cost, right_cost)

        # the heart of this calculator:
        info_gain = current_entropy - (left_ratio * left_entropy + right_ratio * right_entropy)

        return info_gain

    # helper functions:

    def calc_cost_entropy(self, examples: ndarray):
        """
        calculating the entropy of the given examples with regards to costs

        :param examples: the given examples

        :return: tuple((1), (2)):
                 (1): the total cost
                 (2): the entropy of the group of examples (float)
        """
        sick_examples = select_sick_examples(examples)

        num_examples = len(examples)
        num_sick_examples = len(sick_examples)
        num_healthy_examples = num_examples - num_sick_examples

        total_cost, sick_cost, healthy_cost = self.calc_costs(num_sick_examples, num_healthy_examples)

        if are_equal_or_complement(num_examples, num_sick_examples):
            return total_cost, NO_INFO_GAIN

        probabilities = self.calc_probabilities(sick_cost, healthy_cost)

        entropy = self.evaluate_entropy(probabilities)

        return total_cost, entropy

    @staticmethod
    def evaluate_entropy(probabilities: Tuple[float, float]):
        """
        evaluating the entropy of the given probabilities
        usual usage is probabilities=[probability_sick, probability_healthy]

        :param probabilities: the probabilities

        :return: the entropy of the given probabilities (float)
        """
        cumulated_entropy = 0

        # todo remove
        assert sum(probabilities) == 1

        for probability in probabilities:
            if not is_within_floating_point_error_range(probability):
                cumulated_entropy -= probability * log2(probability)

        return cumulated_entropy

    # todo: document - important
    @staticmethod
    def calc_probabilities(sick_cost: int, healthy_cost: int):
        total_cost = sick_cost + healthy_cost

        sick_probability = sick_cost / total_cost
        healthy_probability = healthy_cost / total_cost

        # todo remove
        assert sick_probability + healthy_probability == 1

        return sick_probability, healthy_probability

    # todo: document - important
    @staticmethod
    def calc_ratios(left_cost: int, right_cost: int):
        total_cost = left_cost + right_cost
        
        left_ratio = left_cost / total_cost
        right_ratio = right_cost / total_cost

        return left_ratio, right_ratio

    # todo: document
    @staticmethod
    def calc_costs(num_sick_examples: int, num_healthy_examples: int):
        """
        a simple implementation mainly used to be overridden in ex.4
        assuming each example's cost is equal to each other and to 1

        :param num_sick_examples: the number of sick examples
        :param num_healthy_examples: the number of healthy examples

        :return: tuple((1), (2), (3)):
                 (1): the total cost
                 (2): the sick cost
                 (3): the healthy cost
        """
        total_cost = num_sick_examples + num_healthy_examples

        return total_cost, num_sick_examples, num_healthy_examples
