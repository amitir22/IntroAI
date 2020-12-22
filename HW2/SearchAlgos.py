"""Search Algos: MiniMax, AlphaBeta
"""
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
#TODO: you can import more modules, if needed


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

        # todo: check if target state
        # todo: handle target state (leaf) and calc heuristics.

        # todo: check if leaf
        # todo: handle leaf (game not ended yet but depth=0 reached)

        # todo: assert non leaf and non target state.
        # todo: handle general case where 4 moves available etc...

        raise NotImplementedError


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
        #TODO: erase the following line and implement this function.
        raise NotImplementedError
