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
        if self.goal(state):
            # TODO: make sure the criteria is ok
            # score of goal is the difference in scores
            return state.my_score - state.rival_score, state.direction_from_previous_state
        elif depth == 0:
            return self.utility(state), state.direction_from_previous_state
        else:
            if maximizing_player:
                current_best_value = -np.inf
            else:
                current_best_value = np.inf

            current_best_direction = (0, 0)

            for succ_state in self.succ(state):
                value, direction = self.search(succ_state, depth - 1, not maximizing_player)

                is_max_state_and_new_max_value = maximizing_player and value > current_best_value
                is_min_state_and_new_min_value = not maximizing_player and value < current_best_value

                if is_max_state_and_new_max_value or is_min_state_and_new_min_value:
                    current_best_value = value
                    current_best_direction = direction

            return current_best_value, current_best_direction


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
