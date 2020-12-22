"""
MiniMax Player
"""
from typing import Tuple, List

from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import MiniMax
import numpy as np
import dataclasses
import copy
# TODO: you can import more modules, if needed


@dataclasses.dataclass(init=True, frozen=True)
class PlayerState:
    game_time: float
    penalty_score: int
    board: np.array  # matrix
    num_rows: int
    num_cols: int
    fruit_locations: List[Tuple[int, int]]
    fruits_turns_to_live: int
    my_loc: Tuple[int, int]
    my_score: float
    rival_loc: Tuple[int, int]
    rival_score: float


class Player(AbstractPlayer):
    game_time: float
    penalty_score: int
    search_algo: MiniMax
    current_player_state: PlayerState

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.game_time = game_time
        self.penalty_score = penalty_score
        self.search_algo = MiniMax(None, None, None, None)  # TODO:

    def set_game_params(self, board: np.array):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            :param board: np.array, a 2D matrix of the board.
        output:
            :return: None
        """
        num_rows = len(board)
        num_cols = len(board[0])

        my_loc = [(i, j) for i in range(num_rows) for j in range(num_cols) if board[i][j] == 1][0]
        rival_loc = [(i, j) for i in range(num_rows) for j in range(num_cols) if board[i][j] == 2][0]

        self.current_player_state = PlayerState(game_time=self.game_time, penalty_score=self.penalty_score,
                                                board=board, num_rows=num_rows, num_cols=num_cols,
                                                fruit_locations=[], fruits_turns_to_live=min(num_cols, num_rows),
                                                my_loc=my_loc, my_score=0, rival_loc=rival_loc, rival_score=0)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            :param time_limit: float, time limit for a single turn.
            :param players_score: list:[score player1, score player2]
        output:
            :return: direction: tuple, specifying the Player's movement, chosen from self.directions
        """
        player_state = copy.deepcopy(self)

        raise NotImplementedError

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            :param pos: tuple, the new position of the rival.
        output:
            :return: None
        """
        #TODO: erase the following line and implement this function.
        raise NotImplementedError

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            :param fruits_on_board_dict: dict of {pos: value}
                                         where 'pos' is a tuple describing the fruit's position on board,
                                         'value' is the value of this fruit.
        output:
            :return: None
        """
        #TODO: erase the following line and implement this function. In case you choose not to use it, use 'pass' instead of the following line.
        raise NotImplementedError

    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed

    ########## helper functions for MiniMax algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm

    @staticmethod
    def heuristic_function(state):
        pass

