"""
MiniMax Player
"""
from typing import Tuple, List, Dict

from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import MiniMax
import time
import numpy as np
import dataclasses
import copy
import utils
# TODO: you can import more modules, if needed


# TODO: consider storing in the state class a miniboard instead.
@dataclasses.dataclass(init=True)
class PlayerState:
    MY_TURN = 0
    RIVAL_TURN = 1

    turn: int
    board: np.array  # matrix
    fruit_locations: Dict[Tuple[int, int]: int]  # key: location, value: fruit_score
    fruits_turns_to_live: int
    players_locations: Dict[int: Tuple[int, int]]
    players_scores: Dict[int: int]
    player: AbstractPlayer

    @property
    def my_loc(self):
        return self.players_locations[self.MY_TURN]

    @my_loc.setter
    def my_loc(self, value: Tuple[int, int]):
        self.players_locations[self.MY_TURN] = value

    @property
    def rival_loc(self):
        return self.players_locations[self.RIVAL_TURN]

    @rival_loc.setter
    def rival_loc(self, value: Tuple[int, int]):
        self.players_locations[self.RIVAL_TURN] = value

    @property
    def my_score(self):
        return self.players_scores[self.MY_TURN]

    @my_score.setter
    def my_score(self, value: int):
        self.players_scores[self.MY_TURN] = value

    @property
    def rival_score(self):
        return self.players_scores[self.RIVAL_TURN]

    @rival_score.setter
    def rival_score(self, value: int):
        self.players_scores[self.RIVAL_TURN] = value

    def duplicate(self):
        return copy.deepcopy(self)


class Player(AbstractPlayer):
    BRANCHING_FACTOR = 4
    BLOCK_CELL = -1
    P1_CELL = 1
    P2_CELL = 2

    game_time: float
    penalty_score: int
    search_algo: MiniMax
    current_state: PlayerState
    num_rows: int
    num_cols: int
    directions: List[Tuple[int, int]]
    my_state_list: List[PlayerState]  # TODO: remove before submission. here for debugging purposes.

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.game_time = game_time
        self.penalty_score = penalty_score
        self.search_algo = MiniMax(Player.is_move_valid, Player.successor_function,
                                   None, Player.is_goal)  # TODO: replace None?
        self.directions = utils.get_directions()
        self.my_state_list = []

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

        my_loc = [(i, j) for i in range(num_rows) for j in range(num_cols) if board[i][j] == self.P1_CELL][0]
        rival_loc = [(i, j) for i in range(num_rows) for j in range(num_cols) if board[i][j] == self.P2_CELL][0]

        assert None not in [my_loc, rival_loc]

        board[my_loc] = Player.BLOCK_CELL
        board[rival_loc] = Player.BLOCK_CELL

        players_scores = {PlayerState.MY_TURN: 0, PlayerState.RIVAL_TURN: 0}
        players_locations = {PlayerState.MY_TURN: my_loc, PlayerState.RIVAL_TURN: rival_loc}

        self.current_state = PlayerState(board=board, fruit_locations={}, fruits_turns_to_live=min(num_cols, num_rows),
                                         players_scores=players_scores, players_locations=players_locations,
                                         player=self, turn=PlayerState.MY_TURN)
        self.my_state_list.append(self.current_state)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            :param time_limit: float, time limit for a single turn.
            :param players_score: list:[score player1, score player2]
        output:
            :return: direction: tuple, specifying the Player's movement, chosen from self.directions
        """
        # updating scores
        self.current_state.my_score, self.current_state.rival_score = players_score

        # setting up variables for the anytime-minimax
        should_continue_to_next_iteration = True
        current_depth = 1
        last_minimax_value = 0
        last_best_move = (0, 0)
        time_left = time_limit

        # executing minimax in anytime-contact
        while should_continue_to_next_iteration:
            tick = time.time()
            last_minimax_value, last_best_move = self.search_algo.search(self.current_state, current_depth, True)
            current_depth += 1
            tock = time.time()

            # time management
            time_diff = tock - tick
            time_left -= time_diff

            is_there_no_time_for_next_iteration = time_left < self.BRANCHING_FACTOR * time_diff

            if is_there_no_time_for_next_iteration:
                should_continue_to_next_iteration = False

        # updating next state and returning the next move
        self.current_state.fruits_turns_to_live -= 1

        next_state = self.current_state.duplicate()

        next_state.my_loc = utils.tup_add(self.current_state.my_loc, last_best_move)
        next_state.board[next_state.my_loc] = Player.BLOCK_CELL

        self.current_state = next_state
        self.my_state_list.append(self.current_state)

        return last_best_move

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            :param pos: tuple, the new position of the rival.
        output:
            :return: None
        """
        self.current_state.board[pos] = Player.BLOCK_CELL
        self.current_state.rival_loc = pos

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            :param fruits_on_board_dict: dict of {pos: value}
                                         where 'pos' is a tuple describing the fruit's position on board,
                                         'value' is the value of this fruit.
        output:
            :return: None
        """
        self.current_state.fruit_locations = fruits_on_board_dict

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm

    @staticmethod
    def is_move_valid(board: np.array, location: tuple, direction: tuple):
        new_row, new_col = utils.tup_add(location, direction)

        is_new_loc_in_board = 0 <= new_row <= Player.num_rows and 0 <= new_col <= Player.num_cols

        if is_new_loc_in_board:
            is_not_blocked = board[new_row][new_col] != Player.BLOCK_CELL

            return is_not_blocked
        else:
            return False

    @staticmethod
    def is_goal(state: PlayerState):
        directions = state.player.directions
        am_i_blocked = True
        is_rival_blocked = True

        # checking if i'm blocked
        for direction in directions:
            if am_i_blocked and Player.is_move_valid(state.board, state.my_loc, direction):
                am_i_blocked = False

            if is_rival_blocked and Player.is_move_valid(state.board, state.rival_loc, direction):
                is_rival_blocked = False

        return am_i_blocked or is_rival_blocked

    @staticmethod
    def successor_function(state: PlayerState):
        for direction in state.player.directions:
            next_state = state.duplicate()

            next_state.turn = 1 - state.turn  # if rival -> next=me | else next=rival

            next_state.fruits_turns_to_live -= 1

            loc = state.players_locations[next_state.turn]

            if Player.is_move_valid(state.board, loc, direction):
                new_loc = utils.tup_add(loc, direction)

                if new_loc in state.fruit_locations:
                    del next_state.fruit_locations[new_loc]

                    next_state.players_scores[next_state.turn] += next_state.board[new_loc]

                next_state.board[new_loc] = Player.BLOCK_CELL

                yield next_state
