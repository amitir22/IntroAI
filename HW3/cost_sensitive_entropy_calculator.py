from entropy_calculator import EntropyCalculator
from utilities import FALSE_NEGATIVE_COST_FACTOR


class CostSensitiveEntropyCalculator(EntropyCalculator):
    """
    an EntropyCalculator implementation with consideration to the loss function described in ex.4
    """

    @staticmethod
    def calc_costs(num_sick_examples: int, num_healthy_examples: int):
        """
        calculating costs with consideration to the loss function described in ex.4

        assuming:
        each sick example's cost is DEFAULT_COST_MAJORITY_FACTOR
        each healthy example's cost is 1

        :param num_sick_examples: the number of sick examples
        :param num_healthy_examples: the number of healthy examples

        :return: tuple((1), (2), (3)):
                 (1): the total cost
                 (2): the sick cost
                 (3): the healthy cost
        """
        sick_cost = num_sick_examples * FALSE_NEGATIVE_COST_FACTOR
        healthy_cost = num_healthy_examples

        total_cost = sick_cost + healthy_cost

        return total_cost, sick_cost, healthy_cost
