# List of todos for entire HW3
# TODO: eliminate all todos and asserts
# todo: make sure to document all important functions
# TODO: reorganize functions by some order (especially in utilities.py)
# todo: ...

from pandas import DataFrame
from matplotlib import pyplot

from ImprovedKNNForest import ex7
from KNNForest import ex6
from CostSensitiveID3 import ex4
from ID3 import ex1, ex3
from data_set_handler import DataSetHandler


def main():
    data_handler = DataSetHandler()

    # # todo: remove
    # specific_experiment(data_handler)

    ex1(data_handler)
    # ex3(data_handler)
    ex4(data_handler)
    ex6(data_handler)
    ex7(data_handler)


if __name__ == '__main__':
    main()
