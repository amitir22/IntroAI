from abc import ABC


class InfoGainCalculator(ABC):
    """
    abstract class for information-gain calculators used by Intro to AI course ML models
    """
    def calc_info_gain(self, examples, feature_index, additional_info):
        pass
