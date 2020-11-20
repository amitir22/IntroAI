import random


class MyClass(object):
    def __init__(self):
        print("this is my class")


class Point(object):
    def __init__(self, x=0, y=0):
        print("i am in __init__")
        self.x = x
        self.y = y

    def show(self):
        print("my location is %d,%d" % (self.x, self.y))

    def __str__(self):
        print("i am in __str__")
        return f'({self.x}, {self.y})'


def ex5(num):
    rand_num = random.randint(1, 1001)

    if rand_num > num:
        return 0
    return rand_num


def ex6_extra(num_list, num):
    return random.sample(num_list, num)


def ex7(some_string="-123.321"):
    splitted = some_string.split(".")
    x = int(splitted[0])
    y = int(splitted[1])

    return x + y
