import partB


def test_MyClass():
    lol = partB.MyClass()


def test_Point():
    test_empty_point = partB.Point()
    test_point = partB.Point(1, 2)

    test_empty_point.show()
    test_point.show()


def test_ex5():
    print(partB.ex5(500))
    print(partB.ex5(500))
    print(partB.ex5(500))


def test_ex6():
    print(partB.ex6_extra([1, 2, 3, 4, 5], 3))
    print(partB.ex6_extra([1, 2, 3, 4, 5], 3))
    print(partB.ex6_extra([1, 2, 3, 4, 5], 3))


def test_ex7():
    print(partB.ex7())


def test_partB():
    test_MyClass()
    test_Point()
    test_ex5()
    test_ex6()
    test_ex7()


def main():
    # test_partB()
    p = partB.Point(3, 4)

    print(str(p))


if __name__ == '__main__':
    main()
