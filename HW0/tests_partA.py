import partA


def test_args_less():
    partA.main()
    partA.func4()
    partA.func5()
    partA.func6()
    partA.func7()
    partA.func9()


def test_func2():
    did_fail = False

    if partA.func2(True) != 1 or partA.func2(False) != -1:
        did_fail = True

    if did_fail:
        print("func2 failed")
    else:
        print("func2 success")


def test_func3():
    if partA.func3(2, 3) == 8:
        print("func3 success")
    else:
        print("func3 fail")


def test_func8():
    test_list = [1, 2, 3, 4, 5]

    test_result = partA.func8(test_list)

    if 15 == test_result:
        print("func8 success")
    else:
        print("func8 fail")


def test_func10():
    if partA.func10(6) and not partA.func10(5):
        print("func10 success")
    else:
        print("func10 fail")


def test_partA():
    test_args_less()
    test_func2()
    test_func3()
    test_func8()
    test_func10()


def main():
    test_partA()


if __name__ == '__main__':
    main()
