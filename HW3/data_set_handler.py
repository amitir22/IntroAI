import pandas


class DataSetHandler:
    train_path: str
    test_path: str

    def __init__(self,
                 train_path: str = 'data_sets/train.csv',
                 test_path: str = 'prediction_tests/test.csv'):
        self.train_path = train_path
        self.test_path = test_path

    def read_train_data(self):
        return pandas.read_csv(self.train_path)

    def read_test_data(self):
        return pandas.read_csv(self.test_path)
