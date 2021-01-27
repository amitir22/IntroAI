# List of todos for entire HW3
# TODO: eliminate all todos and asserts
# todo: make sure to document all important functions
# TODO: reorganize functions by some order (especially in utilities.py)
# todo: ...

from numpy import indices
from pandas import DataFrame
from matplotlib import pyplot

from ID3 import ex1, experiment as ex3
from CostSensitiveID3 import ex4
from KNNForest import ex6
from data_set_handler import DataSetHandler


def plot_graphs_of_pair_of_features(data_set: DataFrame, column1: str, column2: str):
    pyplot.figure(f'{column2}[{data_set.columns.tolist().index(column2)}]')

    colors = {'B': 'blue', 'M': 'red'}

    pyplot.xlabel(f'feature: {column1}')
    pyplot.ylabel(f'feature: {column2}')
    pyplot.scatter(data_set[column1], data_set[column2], c=data_set['diagnosis'].map(colors))


def cluster_experiment(data_set_handler: DataSetHandler):
    train_data = data_set_handler.read_train_data()

    # todo:
    column0 = train_data.columns[0]
    column1 = train_data.columns[1]

    for column2 in train_data.columns:
        if column2 not in [column0, column1]:
            plot_graphs_of_pair_of_features(train_data, column1, column2)

    pyplot.show()

    exit(0)


def specific_experiment(data_set_handler: DataSetHandler):
    train_data = data_set_handler.read_train_data()

    column_indices = [12, 16]
    filtered_columns = [train_data.columns[c_index] for c_index in column_indices]

    # todo:
    column0 = train_data.columns[0]
    column1 = train_data.columns[29]

    for column2 in filtered_columns:
        if column2 not in [column0, column1]:
            plot_graphs_of_pair_of_features(train_data, column1, column2)

    pyplot.show()

    exit(0)


def main():
    data_handler = DataSetHandler()

    # # todo: remove
    # specific_experiment(data_handler)

    ex1(data_handler)
    ex3(data_handler)
    ex4(data_handler)
    # ex6(data_handler)


if __name__ == '__main__':
    main()
