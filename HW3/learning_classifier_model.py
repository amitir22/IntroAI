from abc import ABC
from pandas import DataFrame


class LearningClassifierModel(ABC):
    def train(self, dataset: DataFrame):
        pass

    def test(self, dataset: DataFrame):
        pass
