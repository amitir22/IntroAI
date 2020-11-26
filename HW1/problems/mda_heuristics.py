import numpy as np
import networkx as nx
from typing import *

from framework import *
from .mda_problem import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator


__all__ = ['MDAMaxAirDistHeuristic', 'MDASumAirDistHeuristic',
           'MDAMSTAirDistHeuristic', 'MDATestsTravelDistToNearestLabHeuristic']


class MDAMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Max-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This method calculated a lower bound of the distance of the remaining path of the ambulance,
         by calculating the maximum distance within the group of air distances between each two
         junctions in the remaining ambulance path. We don't consider laboratories here because we
         do not know what laboratories would be visited in an optimal solution.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        total_distance_lower_bound = max(self.cached_air_distance_calculator.get_air_distance_between_junctions(j1, j2)
                                         for j1 in all_certain_junctions_in_remaining_ambulance_path
                                         for j2 in all_certain_junctions_in_remaining_ambulance_path
                                         if j1.index < j2.index)

        return total_distance_lower_bound


class MDASumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Sum-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDASumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic evaluates the distance of the remaining ambulance route in the following way:
        It builds a path that starts in the current ambulance's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all certain junctions (in `all_certain_junctions_in_remaining_ambulance_path`) that haven't
         been visited yet.
        The remaining distance estimation is the cost of this built path.
        Note that we ignore here the problem constraints (like enforcing the #matoshim and free
         space in the ambulance's fridge). We only make sure to visit all certain junctions in
         `all_certain_junctions_in_remaining_ambulance_path`.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)

        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        air_distance_function = self.cached_air_distance_calculator.get_air_distance_between_junctions

        all_remaining_junctions = all_certain_junctions_in_remaining_ambulance_path

        current_junction = state.current_location
        sum_air_dist_cost = 0.0

        while len(all_remaining_junctions) > 1:
            all_remaining_junctions.remove(current_junction)

            current_minimum_distance = float('inf')
            current_next_junction = all_remaining_junctions[0]
            current_minimum_score = (current_minimum_distance, current_next_junction.index)

            for candidate_next_junction in all_remaining_junctions:
                candidate_distance = air_distance_function(current_junction, candidate_next_junction)

                candidate_score = (candidate_distance, candidate_next_junction.index)

                if candidate_score < current_minimum_score:
                    current_minimum_distance = candidate_distance
                    current_next_junction = candidate_next_junction
                    current_minimum_score = (current_minimum_distance, current_next_junction.index)

            sum_air_dist_cost += current_minimum_distance
            current_junction = current_next_junction

        return sum_air_dist_cost


class MDAMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-MST-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound for the distance of the remaining route of the ambulance.
        Here this remaining distance is bounded (from below) by the weight of the minimum-spanning-tree
         of the graph, in-which the vertices are the junctions in the remaining ambulance route, and the
         edges weights (edge between each junctions pair) are the air-distances between the junctions.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        return self._calculate_junctions_mst_weight_using_air_distance(
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state))

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: List[Junction]) -> float:
        """
              Use `networkx` (nx) package (already imported in this file) to calculate the weight
               of the minimum-spanning-tree of the graph in which the vertices are the given junctions
               and there is an edge between each pair of distinct junctions (no self-loops) for which
               the weight is the air distance between these junctions.
              Use the method `self.cached_air_distance_calculator.get_air_distance_between_junctions()`
               to calculate the air distance between the two junctions.
              Use `nx.minimum_spanning_tree()` to get an MST. Calculate the MST size using the method
              `.size(weight='weight')`. Do not manually sum the edges' weights.
        """
        get_air_distance = self.cached_air_distance_calculator.get_air_distance_between_junctions

        # weighted edges format for edge "u-v" with weight=w : (u, v, w)
        G = nx.Graph()
        G.add_nodes_from(junctions)
        G.add_weighted_edges_from([(j1, j2, get_air_distance(j1, j2))
                                   for j1 in junctions
                                   for j2 in junctions
                                   if j1.index < j2.index])

        mst = nx.minimum_spanning_tree(G)
        return mst.size(weight='weight')


class MDATestsTravelDistToNearestLabHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-TimeObjectiveSumOfMinAirDistFromLab'

    def __init__(self, problem: GraphProblem):
        super(MDATestsTravelDistToNearestLabHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound to the remained tests-travel-distance of the remained ambulance path.
        The main observation is that driving from a laboratory to a reported-apartment does not increase the
         tests-travel-distance cost. So the best case (lowest cost) is when we go to the closest laboratory right
         after visiting any reported-apartment.
        If the ambulance currently stores tests, this total remained cost includes the #tests_on_ambulance times
         the distance from the current ambulance location to the closest lab.
        The rest part of the total remained cost includes the distance between each non-visited reported-apartment
         and the closest lab (to this apartment) times the roommates in this apartment (as we take tests for all
         roommates).
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        def air_dist_to_closest_lab(junction: Junction) -> float:
            """
            Returns the distance between `junction` and the laboratory that is closest to `junction`.
            """
            assert isinstance(self.problem, MDAProblem)
            return min(self.cached_air_distance_calculator.get_air_distance_between_junctions(junction, lab.location)
                       for lab in self.problem.problem_input.laboratories)

        previous_apartments_tests_that_are_still_on_ambulance = list(state.tests_on_ambulance)
        num_of_tests_on_ambulance = sum(apartment.nr_roommates
                                        for apartment in previous_apartments_tests_that_are_still_on_ambulance)

        previous_cost = num_of_tests_on_ambulance * air_dist_to_closest_lab(state.current_location)

        remaining_apartments = self.problem.get_reported_apartments_waiting_to_visit(state)

        remaining_cost = sum(apartment.nr_roommates * air_dist_to_closest_lab(apartment.location)
                             for apartment in remaining_apartments)

        total_cost = previous_cost + remaining_cost

        return total_cost
