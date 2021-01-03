"""
MiniMax Player with AlphaBeta pruning
"""
from players.AbstractPlayer import AbstractPlayer
from typing import Tuple, List
from SearchAlgos import AlphaBeta
import time
import numpy as np
import utils
from utils import PlayerState, BRANCHING_FACTOR, BLOCK_CELL, P1_CELL, P2_CELL, MY_TURN, RIVAL_TURN


class Player(AbstractPlayer):
    game_time: float
    penalty_score: int
    search_algo: AlphaBeta
    current_state: PlayerState
    num_rows: int
    num_cols: int
    directions: List[Tuple[int, int]]

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.search_algo = AlphaBeta(utils.heuristic_function, utils.successor_states_of, utils.perform_move)
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
        self.num_rows = len(board)
        self.num_cols = len(board[0])

        my_loc = np.where(board == P1_CELL)
        my_loc = my_loc[0][0], my_loc[1][0]
        rival_loc = np.where(board == P2_CELL)
        rival_loc = rival_loc[0][0], rival_loc[1][0]

        assert None not in [my_loc, rival_loc]

        board[my_loc] = BLOCK_CELL
        board[rival_loc] = BLOCK_CELL

        players_scores = {MY_TURN: 0, RIVAL_TURN: 0}
        players_locations = {MY_TURN: my_loc, RIVAL_TURN: rival_loc}

        fruits_turns_to_live = min(self.num_rows, self.num_cols)

        self.current_state = PlayerState(board=board, fruit_locations={}, fruits_turns_to_live=fruits_turns_to_live,
                                         players_scores=players_scores, players_locations=players_locations,
                                         player=self, turn=MY_TURN, penalty_score=self.penalty_score)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            :param time_limit: float, time limit for a single turn.
            :param players_score: list:[score player1, score player2]
        output:
            :return: direction: tuple, specifying the Player's movement, chosen from self.directions
        """
        # starting the clock
        tick = time.time()

        # updating scores
        self.current_state.my_score, self.current_state.rival_score = players_score
        self.current_state.turn = MY_TURN

        # setting up variables for the anytime-minimax
        should_continue_to_next_iteration = True
        current_depth = 0
        last_minimax_value = 0
        last_best_move = (0, 0)
        time_left = time_limit

        num_blocks_on_board = len(np.where(self.current_state.board == BLOCK_CELL))
        num_zeros_on_board = self.num_rows * self.num_cols - num_blocks_on_board

        # executing minimax in anytime-contact
        while should_continue_to_next_iteration:
            current_depth += 1
            last_minimax_value, last_best_move = self.search_algo.search(self.current_state, current_depth, True)

            # time management
            tock = time.time()
            time_diff = tock - tick
            time_left -= time_diff
            tick = time.time()

            is_there_no_time_for_next_iteration = time_left < BRANCHING_FACTOR * time_diff or \
                                                  time_left <= 0.1 * time_limit

            is_depth_covers_all_cells = current_depth >= num_zeros_on_board

            if is_there_no_time_for_next_iteration or is_depth_covers_all_cells:
                should_continue_to_next_iteration = False

        # updating next state and returning the next move
        self.current_state.fruits_turns_to_live -= 1
        next_location = utils.tup_add(self.current_state.my_loc, last_best_move)

        assert last_best_move != (0, 0)

        # updating the board in the next state and moving the current_state to point at the next
        self.current_state = utils.perform_move(self.current_state, MY_TURN, next_location)

        # todo: remove
        print('alpha-beta:')
        print(f'turn: {len(self.my_state_list)}, depth: {current_depth}, minimax value: {last_minimax_value} '
              f'scores: player1: {self.current_state.my_score} ,player2: {self.current_state.rival_score}')

        return last_best_move

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            :param pos: tuple, the new position of the rival.
        output:
            :return: None
        """
        self.current_state = utils.perform_move(self.current_state, RIVAL_TURN, pos)

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
