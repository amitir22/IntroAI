import operator
import numpy as np
import os
import dataclasses
import copy
import players.AbstractPlayer
from typing import Tuple, Dict

ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf
BRANCHING_FACTOR = 4
BLOCK_CELL = -1
P1_CELL = 1
P2_CELL = 2
MY_TURN = 0
RIVAL_TURN = 1


@dataclasses.dataclass(init=True)
class PlayerState:
    turn: int
    board: np.array  # matrix
    fruit_locations: Dict[Tuple[int, int], int]  # key: location, value: fruit_score
    fruits_turns_to_live: int
    players_locations: Dict[int, Tuple[int, int]]
    players_scores: Dict[int, int]
    player: players.AbstractPlayer

    @property
    def my_loc(self):
        return self.players_locations[MY_TURN]

    @my_loc.setter
    def my_loc(self, value: Tuple[int, int]):
        self.players_locations[MY_TURN] = value

    @property
    def rival_loc(self):
        return self.players_locations[RIVAL_TURN]

    @rival_loc.setter
    def rival_loc(self, value: Tuple[int, int]):
        self.players_locations[RIVAL_TURN] = value

    @property
    def my_score(self):
        return self.players_scores[MY_TURN]

    @my_score.setter
    def my_score(self, value: int):
        self.players_scores[MY_TURN] = value

    @property
    def rival_score(self):
        return self.players_scores[RIVAL_TURN]

    @rival_score.setter
    def rival_score(self, value: int):
        self.players_scores[RIVAL_TURN] = value

    def duplicate(self):
        return copy.deepcopy(self)


def get_directions():
    """Returns all the possible directions of a player in the game as a list of tuples.
    """
    return [(1, 0), (0, 1), (-1, 0), (0, -1)]


def tup_add(t1, t2):
    """
    returns the sum of two tuples as tuple.
    """
    return tuple(map(operator.add, t1, t2))


def get_board_from_csv(board_file_name):
    """Returns the board data that is saved as a csv file in 'boards' folder.
    The board data is a list that contains: 
        [0] size of board
        [1] blocked poses on board
        [2] starts poses of the players
    """
    board_path = os.path.join('boards', board_file_name)
    board = np.loadtxt(open(board_path, "rb"), delimiter=" ")
    
    # mirror board
    board = np.flipud(board)
    i, j = len(board), len(board[0])
    blocks = np.where(board == -1)
    blocks = [(blocks[0][i], blocks[1][i]) for i in range(len(blocks[0]))]
    start_player_1 = np.where(board == 1)
    start_player_2 = np.where(board == 2)
    
    if len(start_player_1[0]) != 1 or len(start_player_2[0]) != 1:
        raise Exception('The given board is not legal - too many start locations.')
    
    start_player_1 = (start_player_1[0][0], start_player_1[1][0])
    start_player_2 = (start_player_2[0][0], start_player_2[1][0])

    return [(i, j), blocks, [start_player_1, start_player_2]]
