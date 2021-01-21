from numpy import ndarray, log2
from info_gain_calculator import InfoGainCalculator
import utilities


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

        if utilities.is_homogenous(num_examples, num_left_examples):  # here to save some calculation time
            return utilities.DEFAULT_INFO_GAIN

        right_examples = examples[examples[:, feature_index] >= mean_feature_value]
        num_right_examples = len(right_examples)

        left_ratio = num_left_examples / num_examples
        right_ratio = num_right_examples / num_examples

        current_entropy = self.calc_entropy(examples)
        left_entropy = left_ratio * self.calc_entropy(left_examples)
        right_entropy = right_ratio * self.calc_entropy(right_examples)

        # the heart of this calculator:
        info_gain = current_entropy - (left_entropy + right_entropy)

        return info_gain

    # helper functions:

    @staticmethod
    def calc_entropy(examples: ndarray):
        """
        calculating the entropy of the given examples

        :param examples: the given examples

        :return: the entropy of the group of examples (float)
        """
        num_examples = len(examples)
        num_sick_examples = len(utilities.select_sick_examples(examples))

        # TODO: remove
        assert num_examples != 0

        if utilities.is_homogenous(num_examples, num_sick_examples):
            return utilities.DEFAULT_INFO_GAIN

        num_healthy_examples = num_examples - num_sick_examples

        probability_healthy = num_healthy_examples / num_examples
        probability_sick = num_sick_examples / num_examples

        entropy_healthy = -probability_healthy * log2(probability_healthy)
        entropy_sick = -probability_sick * log2(probability_sick)

        return entropy_healthy + entropy_sick
