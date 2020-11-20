import functools


def main():
    print("hello world")


def func2(some_bool):
    if some_bool:
        return 1
    return -1


def func3(x, y):
    return x ** y


def func4():
    x = 123
    y = 321
    x, y = y, x
    print((x, y))


def func5():
    my_list = []
    my_list += [1, 2]
    my_list = my_list[::-1]
    print(my_list)


def func6():
    for i in range(2, 23 + 1):
        print(i)


def func7():
    num_list = list(range(1, 11))
    print(num_list[3::2])


def func8(num_list):
    return functools.reduce(lambda s, member: s + member, num_list)


def func9():
    with open('my_file.txt', 'w') as out_file:
        out_file.write("i know how to write")


def func10(num):
    return num == sum([i for i in range(1, num) if num % i == 0])


def func8_extra(num_list):
    print(sum(num_list))


if __name__ == "__main__":
    main()