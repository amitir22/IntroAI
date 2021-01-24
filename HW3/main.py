# List of todos for entire HW3
# TODO: eliminate all todos
# todo: make sure to document all important functions
# TODO: reorganize functions by some order (especially in utilities.py)
# todo: ...

from ID3 import ex1, experiment as ex3
from data_set_handler import DataSetHandler


def main():
    data_handler = DataSetHandler()
    ex1(data_handler)
    ex3(data_handler)


if __name__ == '__main__':
    main()
