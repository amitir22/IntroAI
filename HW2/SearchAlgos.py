"""Search Algos: MiniMax, AlphaBeta
"""
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
import numpy as np


# TODO: you can import more modules, if needed


class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move

        # TODO: make sure if need to delete
        self.goal = goal

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player):
        """Start the MiniMax algorithm.
        input:
            :param state: The state to start from.
            :param depth: The maximum allowed depth for the algorithm.
            :param maximizing_player: Whether this is a max node (True) or a min node (False).
        output:
            :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        assert depth > 0
        heuristic_function = self.utility
        succ_states = self.succ(state)

        if len(succ_states) == 0:  # goal state
            prize = 1000000

            if maximizing_player:
                score_diff = state.my_score - state.rival_score - state.player.penalty_score
            else:
                score_diff = state.my_score - state.rival_score + state.player.penalty_score

            if score_diff > 0:
                return prize, None
            else:
                return -prize, None

        current_best_direction = (0, 0)

        if maximizing_player:
            current_max_value = -np.inf

            for succ_state, succ_direction in succ_states:
                if depth == 1:  # expanding leaves
                    value = heuristic_function(succ_state)
                else:
                    value, direction = self.search(succ_state, depth - 1, not maximizing_player)

                if value > current_max_value:
                    current_max_value = value
                    current_best_direction = succ_direction

            assert current_best_direction != (0, 0)

            return current_max_value, current_best_direction
        else:
            current_min_value = np.inf

            for succ_state, succ_direction in succ_states:
                if depth == 1:  # expanding leaves
                    value = heuristic_function(succ_state)
                else:
                    value, direction = self.search(succ_state, depth - 1, not maximizing_player)

                if value < current_min_value:
                    current_min_value = value
                    current_best_direction = succ_direction

            assert current_best_direction != (0, 0)

            return current_min_value, current_best_direction


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        input:
            :param state: The state to start from.
            :param depth: The maximum allowed depth for the algorithm.
            :param maximizing_player: Whether this is a max node (True) or a min node (False).
            :param alpha: alpha value
            :param: beta: beta value
        output:
            :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        # TODO: erase the following line and implement this function.
        raise NotImplementedError
