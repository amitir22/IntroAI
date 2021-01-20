import pandas
import utilities
from TDIDTree import TDIDTree


class ID3:
    is_initialized: bool
    decision_tree: TDIDTree

    def train(self, dataset: pandas.DataFrame):
        examples = dataset.to_numpy()

        first_column_index = utilities.STATUS_COLUMN_INDEX + 1
        last_column_index = len(dataset.columns)  # todo: maybe need to add +1?

        features_indexes = list(range(first_column_index, last_column_index))

        self.decision_tree = TDIDTree(examples, features_indexes)
        self.is_initialized = True

    def test(self, dataset: pandas.DataFrame):
        examples = dataset.to_numpy()

        if self.is_initialized:
            return self.decision_tree.classify(examples)
        else:
            raise NotImplementedError('object of type ID3 is not initialized')
