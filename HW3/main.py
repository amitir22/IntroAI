from ImprovedKNNForest import ex7
from KNNForest import ex6
from CostSensitiveID3 import ex4
from ID3 import ex1, ex3
from data_set_handler import DataSetHandler


def main():
    data_handler = DataSetHandler()

    ex1(data_handler)
    ex3(data_handler)
    ex4(data_handler)
    ex6(data_handler)
    ex7(data_handler)


if __name__ == '__main__':
    main()
