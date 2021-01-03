import operator
import numpy as np
import os
import dataclasses
import copy
from typing import Tuple, Dict

ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf
BRANCHING_FACTOR = 4
BLOCK_CELL = -1
P1_CELL = 1
P2_CELL = 2
MY_TURN = 0
RIVAL_TURN = 1


# State model:


@dataclasses.dataclass(init=True)
class PlayerState:
    turn: int
    board: np.array  # matrix
    fruit_locations: Dict[Tuple[int, int], int]  # key: location, value: fruit_score
    fruits_turns_to_live: int
    players_locations: Dict[int, Tuple[int, int]]
    players_scores: Dict[int, int]
    player: object  # supposed to be a Player class but i can't really declare it here

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


# Custom major functions:

def perform_move(state: PlayerState, player_turn: int, target_location: tuple):
    """
    input:
        :param state: the state which we update
        :param player_turn: if 0 -> moving my player, else (=1) -> moving rival player
        :param target_location: the target of the move
    output:
        :return: the updated state. None if the move given was not valid
    """
    next_state = state.duplicate()

    if is_move_valid(next_state.board, target_location):
        if target_location in next_state.fruit_locations and state.fruits_turns_to_live > 0:
            del next_state.fruit_locations[target_location]
            next_state.players_scores[player_turn] += next_state.board[target_location]

        next_state.players_locations[player_turn] = target_location
        next_state.board[target_location] = BLOCK_CELL

        return next_state
    return None


def sorted_successor_states_of(state: PlayerState):
    """
    :param state: the state we expand
    :return: tuple (a successor state, the direction to the successor)
    """
    heuristic_values_of_state = []
    successor_states = []

    for direction in get_directions():
        player_old_loc = state.players_locations[state.turn]
        player_new_loc = tup_add(player_old_loc, direction)

        if is_move_valid(state.board, player_new_loc):
            next_state = perform_move(state, state.turn, player_new_loc)

            next_state.fruits_turns_to_live -= 1
            next_state.turn = 1 - state.turn  # if rival -> next=me | else next=rival

            h_value = heuristic_function(next_state)

            heuristic_values_of_state.append(h_value)
            successor_states.append((next_state, direction))

    sorted_indices = np.argsort(heuristic_values_of_state)
    sorted_successor_states = []

    for index in sorted_indices:
        sorted_successor_states.append(successor_states[index])

    return sorted_successor_states


def heuristic_function(state: PlayerState):
    a = 2  # todo: change

    my_row, my_col = state.my_loc

    score_available_cells = 1
    score_blocked_cells = 1

    for row in range(my_row - a, my_row + a + 1):
        for col in range(my_col - a, my_col + a + 1):
            if (row, col) != (my_row, my_col):
                if is_location_in_board(state.board, (row, col)):
                    cell_value = state.board[(row, col)]

                    if cell_value > 0:
                        score_available_cells += cell_value
                    elif cell_value < 0:
                        score_blocked_cells -= cell_value
                    else:  # cell_value = 0
                        score_available_cells += 1
                else:
                    score_blocked_cells += 1

    current_min_fruit_dist = np.inf
    current_min_fruit_score = 0

    for fruit_loc in state.fruit_locations:
        fruit_score = state.board[fruit_loc]

        my_fruit_dist = m_dist(state.my_loc, fruit_loc)
        rival_fruit_dist = m_dist(state.rival_loc, fruit_loc)

        is_fruit_viable = my_fruit_dist <= rival_fruit_dist and my_fruit_dist < state.fruits_turns_to_live

        if is_fruit_viable and (my_fruit_dist, fruit_score) < (current_min_fruit_dist, current_min_fruit_score):
            current_min_fruit_dist, current_min_fruit_score = my_fruit_dist, fruit_score

    score_closest_fruit_m_dist = current_min_fruit_dist
    score_adversary_m_dist = m_dist(state.my_loc, state.rival_loc) + \
                             np.linalg.norm(np.array(state.my_loc) - np.array(state.rival_loc))

    possible_loc1 = my_row, my_col + 1
    possible_loc2 = my_row, my_col - 1
    possible_loc3 = my_row + 1, my_col
    possible_loc4 = my_row - 1, my_col

    is_not_hole_state = int(is_move_valid(state.board, possible_loc1) or
                            is_move_valid(state.board, possible_loc2) or
                            is_move_valid(state.board, possible_loc3) or
                            is_move_valid(state.board, possible_loc4))

    # todo: check if need to add '+1' in previous players
    return is_not_hole_state * ((score_available_cells / score_blocked_cells) /
                                (score_closest_fruit_m_dist + score_adversary_m_dist + 1))


# Custom helper functions:

def is_move_valid(board: np.array, target_location: tuple):
    if is_location_in_board(board, target_location):
        is_not_blocked = board[target_location] != BLOCK_CELL

        return is_not_blocked
    else:
        return False


def is_location_in_board(board: np.array, location: tuple):
    num_rows = len(board)
    num_cols = len(board[0])

    row, col = location

    return 0 <= row < num_rows and 0 <= col < num_cols


def m_dist(loc1: Tuple[int, int], loc2: Tuple[int, int]):
    row1, col1 = loc1
    row2, col2 = loc2

    return np.abs(row2 - row1) + np.abs(col2 - col1)


def get_directions():
    """Returns all the possible directions of a player in the game as a list of tuples.
    """
    return [(1, 0), (0, 1), (-1, 0), (0, -1)]


# Segel functions:

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
