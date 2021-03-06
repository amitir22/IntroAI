from framework import *
from problems import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union, Optional

# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map.csv"))

# Make sure that the whole execution is deterministic.
# This is important, because we expect to get the exact same results
# in each execution.
Consts.set_seed()


def plot_distance_and_expanded_wrt_weight_figure(
        problem_name: str,
        weights: Union[np.ndarray, List[float]],
        total_cost: Union[np.ndarray, List[float]],
        total_nr_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    """
    weights, total_cost, total_nr_expanded = np.array(weights), np.array(total_cost), np.array(total_nr_expanded)
    assert len(weights) == len(total_cost) == len(total_nr_expanded)
    assert len(weights) > 0
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(weights)

    fig, ax1 = plt.subplots()

    p1, = ax1.plot(weights, total_cost, color='blue', linestyle='solid', label='Solution cost')

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Solution cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    p2, = ax2.plot(weights, total_nr_expanded, color='red', linestyle='solid', label='#Expanded states')

    # ax2: Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('#Expanded states', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_xlabel('weight')

    curves = [p1, p2]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    fig.tight_layout()
    plt.title(f'Quality vs. time for wA* \non problem {problem_name}')
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem, n: int = 30,
                                   max_nr_states_to_expand: Optional[int] = 40_000,
                                   low_heuristic_weight: float = 0.5, high_heuristic_weight: float = 0.95):
    """
    1. Create an array of `n` numbers equally spread in the segment
       [low_heuristic_weight, high_heuristic_weight]
       (including the edges). You can use `np.linspace()` for that.
    2. For each weight in that array run the wA* algorithm, with the
       given `heuristic_type` over the given problem. For each such run,
       if a solution has been found (res.is_solution_found), store the
       cost of the solution (res.solution_g_cost), the number of
       expanded states (res.nr_expanded_states), and the weight that
       has been used in this iteration. Store these in 3 lists (list
       for the costs, list for the #expanded and list for the weights).
       These lists should be of the same size when this operation ends.
       Don't forget to pass `max_nr_states_to_expand` to the AStar c'tor.
    3. Call the function `plot_distance_and_expanded_wrt_weight_figure()`
    """
    #     with these 3 generated lists.
    weight_values = np.linspace(low_heuristic_weight, high_heuristic_weight, n)
    list_of_costs = []
    list_of_num_expanded = []
    list_of_successful_weights = []

    for current_heuristic_weight in weight_values:
        w_A_star = AStar(heuristic_type, current_heuristic_weight)
        res = w_A_star.solve_problem(problem)

        if res.is_solution_found:
            list_of_costs.append(res.solution_g_cost)
            list_of_num_expanded.append(res.nr_expanded_states)
            list_of_successful_weights.append(current_heuristic_weight)

    plot_distance_and_expanded_wrt_weight_figure(problem.name, list_of_successful_weights,
                                                 list_of_costs, list_of_num_expanded)


# --------------------------------------------------------------------
# ------------------------ StreetsMap Problem ------------------------
# --------------------------------------------------------------------

def toy_map_problem_experiments():
    print()
    print('Solve the map problem.')

    # Ex.10
    toy_map_problem = MapProblem(streets_map, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(toy_map_problem)
    print(res)

    # Ex.12
    a_star = AStar(NullHeuristic)
    res = a_star.solve_problem(toy_map_problem)
    print(res)

    # Ex.13
    w_A_star = AStar(AirDistHeuristic)
    res = w_A_star.solve_problem(toy_map_problem)
    print(res)

    # Ex.15
    run_astar_for_weights_in_range(AirDistHeuristic, toy_map_problem)


# --------------------------------------------------------------------
# ---------------------------- MDA Problem ---------------------------
# --------------------------------------------------------------------

loaded_problem_inputs_by_size = {}
loaded_problems_by_size_and_opt_obj = {}


def get_mda_problem(
        problem_input_size: str = 'small',
        optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance):
    if (problem_input_size, optimization_objective) in loaded_problems_by_size_and_opt_obj:
        return loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)]
    assert problem_input_size in {'small', 'moderate', 'big'}
    if problem_input_size not in loaded_problem_inputs_by_size:
        loaded_problem_inputs_by_size[problem_input_size] = MDAProblemInput.load_from_file(
            f'{problem_input_size}_mda.in', streets_map)
    problem = MDAProblem(
        problem_input=loaded_problem_inputs_by_size[problem_input_size],
        streets_map=streets_map,
        optimization_objective=optimization_objective)
    loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)] = problem
    return problem


def basic_mda_problem_experiments():
    print()
    print('Solve the MDA problem (small input, only distance objective, UniformCost).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)

    # Ex.18
    uc = UniformCost()
    res = uc.solve_problem(small_mda_problem_with_distance_cost)
    print(res)


def mda_problem_with_astar_experiments():
    print()
    print('Solve the MDA problem (moderate input, only distance objective, A*, '
          'MaxAirDist & SumAirDist & MSTAirDist heuristics).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.22
    a_star = AStar(MDAMaxAirDistHeuristic)
    res = a_star.solve_problem(moderate_mda_problem_with_distance_cost)

    print(res)

    # Ex.25
    a_star = AStar(MDASumAirDistHeuristic)
    res = a_star.solve_problem(moderate_mda_problem_with_distance_cost)

    print(res)

    # Ex.28
    a_star = AStar(MDAMSTAirDistHeuristic)
    res = a_star.solve_problem(moderate_mda_problem_with_distance_cost)

    print(res)


def mda_problem_with_weighted_astar_experiments():
    print()
    print('Solve the MDA problem (small & moderate input, only distance objective, wA*).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)
    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.30
    run_astar_for_weights_in_range(MDAMSTAirDistHeuristic, small_mda_problem_with_distance_cost)

    # Ex.30
    run_astar_for_weights_in_range(MDASumAirDistHeuristic, moderate_mda_problem_with_distance_cost)


def monetary_cost_objectives_mda_problem_experiments():
    print()
    print('Solve the MDA problem (monetary objectives).')

    small_mda_problem_with_monetary_cost = get_mda_problem('small', MDAOptimizationObjective.Monetary)
    moderate_mda_problem_with_monetary_cost = get_mda_problem('moderate', MDAOptimizationObjective.Monetary)

    # Ex.32
    uc = UniformCost()
    res = uc.solve_problem(small_mda_problem_with_monetary_cost)
    print(res)

    # Ex.32
    uc = UniformCost()
    res = uc.solve_problem(moderate_mda_problem_with_monetary_cost)
    print(res)


def multiple_objectives_mda_problem_experiments():
    print()
    print('Solve the MDA problem (moderate input, distance & tests-travel-distance objectives).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)
    moderate_mda_problem_with_tests_travel_dist_cost = get_mda_problem('moderate', MDAOptimizationObjective.TestsTravelDistance)

    # Ex.35
    a_star = AStar(MDATestsTravelDistToNearestLabHeuristic)
    res = a_star.solve_problem(moderate_mda_problem_with_tests_travel_dist_cost)

    print(res)

    # Ex.38
    a1_star = AStar(MDAMSTAirDistHeuristic)
    res = a1_star.solve_problem(moderate_mda_problem_with_distance_cost)

    optimal_distance_cost = res.solution_g_cost
    eps = 0.6
    max_distance_cost = (1 + eps) * optimal_distance_cost

    # opening every node that his distance cost is lower then the allowed range of (1 + e) * optimal_dist_cost
    node_open_criteria = lambda node: node.cost.distance_cost <= max_distance_cost

    a2_star = AStar(heuristic_function_type=MDATestsTravelDistToNearestLabHeuristic, open_criterion=node_open_criteria)
    res = a2_star.solve_problem(moderate_mda_problem_with_tests_travel_dist_cost)
    print(res)


def mda_problem_with_astar_epsilon_experiments():
    print()
    print('Solve the MDA problem (small input, distance objective, using A*eps, use non-acceptable '
          'heuristic as focal heuristic).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)

    # Firstly solve the problem with AStar & MST heuristic for having a reference for #devs.
    astar = AStar(MDAMSTAirDistHeuristic)
    res = astar.solve_problem(small_mda_problem_with_distance_cost)
    print(res)

    def within_focal_h_sum_priority_function(node: SearchNode, problem: GraphProblem, solver: AStarEpsilon):
        if not hasattr(solver, '__focal_heuristic'):
            setattr(solver, '__focal_heuristic', MDASumAirDistHeuristic(problem=problem))
        focal_heuristic = getattr(solver, '__focal_heuristic')
        return focal_heuristic.estimate(node.state)

    # Ex.43
    # Try using A*eps to improve the speed (#dev) with a non-acceptable heuristic.
    focal_epsilon = 0.23
    max_focal_size = 40
    a_star_epsilon = AStarEpsilon(heuristic_function_type=MDAMSTAirDistHeuristic,
                                  within_focal_priority_function=within_focal_h_sum_priority_function,
                                  focal_epsilon=focal_epsilon,
                                  max_focal_size=max_focal_size)

    res = a_star_epsilon.solve_problem(small_mda_problem_with_distance_cost)
    print(res)


def mda_problem_anytime_astar_experiments():
    print()
    print('Solve the MDA problem (moderate input, only distance objective, Anytime-A*, '
          'MSTAirDist heuristics).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.46
    max_nr_states_to_expand_per_iteration = 1000

    anytime_a_star = AnytimeAStar(heuristic_function_type=MDAMSTAirDistHeuristic,
                                  max_nr_states_to_expand_per_iteration=max_nr_states_to_expand_per_iteration)
    res = anytime_a_star.solve_problem(moderate_mda_problem_with_distance_cost)
    print(res)


def run_all_experiments():
    print('Running all experiments')
    toy_map_problem_experiments()
    basic_mda_problem_experiments()
    mda_problem_with_astar_experiments()
    mda_problem_with_weighted_astar_experiments()
    monetary_cost_objectives_mda_problem_experiments()
    multiple_objectives_mda_problem_experiments()
    mda_problem_with_astar_epsilon_experiments()
    mda_problem_anytime_astar_experiments()


if __name__ == '__main__':
    run_all_experiments()
