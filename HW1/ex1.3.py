from math import factorial as f


def get_hardcoded_input():
    return  [(7, 2, 1, 'seconds'),
             (7, 3, 60, 'minutes'),
             (8, 3, 60, 'hours'),
             (8, 4, 1, 'hours'),
             (9, 3, 24, 'days'),
             (10, 3, 365.0 / 12, 'months'),
             (11, 3, 12, 'years'),
             (12, 3, 1000, 'thousand years'),
             (12, 4, 1, 'thousand years'),
             (13, 4, 1000, 'million years')]


def generate_next_input():
    for k, m, mult, units in get_hardcoded_input():
        yield k, m, mult, units


def calc_num_of_possible_paths(k: int, m: int):
    number_of_houses_permutations = f(k)
    number_of_possibilities_for_last_lab = m
    newtons_binomial_factor = (1 + m) ** k

    num_of_possible_paths = number_of_houses_permutations * \
                            number_of_possibilities_for_last_lab * \
                            newtons_binomial_factor

    return num_of_possible_paths


def calc_num_of_testing_paths_in_1_sec(k: int, m: int):
    return (2 ** 30) / (100 * (k + m))


def calc_estimated_calculation_time(k: int, m: int):
    work = calc_num_of_possible_paths(k, m)
    power = calc_num_of_testing_paths_in_1_sec(k, m)
    time = work / power

    return time


def main():
    current_multiplier = 1 # the first unit is in seconds.
    
    for k, m, multiplier, units in generate_next_input():
        t = calc_estimated_calculation_time(k, m)
        current_multiplier *= multiplier
        
        t_in_units = t / current_multiplier
        
        print(f'k={k}, m={m}, t={t_in_units}[{units}]')


if __name__ == '__main__':
    main()
