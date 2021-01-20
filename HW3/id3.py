import pandas
import utilities
from tdidt import TDIDTree
from feature_selector import FeatureSelector


class ID3:
    is_initialized: bool
    decision_tree: TDIDTree
    feature_selector: FeatureSelector

    def __init__(self, feature_selector: FeatureSelector):
        self.feature_selector = feature_selector

    def train(self, dataset: pandas.DataFrame):
        examples = dataset.to_numpy()

        first_column_index = utilities.STATUS_COLUMN_INDEX + 1
        last_column_index = len(dataset.columns)  # todo: maybe need to add +1? - no because start from 0

        features_indexes = list(range(first_column_index, last_column_index))

        self.decision_tree = TDIDTree(examples, features_indexes, self.feature_selector)
        self.is_initialized = True

    def test(self, dataset: pandas.DataFrame):
        examples = dataset.to_numpy()

        if self.is_initialized:
            return self.decision_tree.classify(examples)
        else:
            raise NotImplementedError('object of type ID3 is not initialized')
