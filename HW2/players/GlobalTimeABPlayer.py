"""
MiniMax Player with AlphaBeta pruning and global time
"""
from players.AbstractPlayer import AbstractPlayer
from typing import Tuple, List, Dict
from SearchAlgos import AlphaBeta
import time
import numpy as np
import utils
from utils import PlayerState, BRANCHING_FACTOR, BLOCK_CELL, P1_CELL, P2_CELL, MY_TURN, RIVAL_TURN


class Player(AbstractPlayer):
    SQRT_2 = 2.0 ** 0.5

    game_time_left: float
    game_time: float
    penalty_score: int
    search_algo: AlphaBeta
    current_state: utils.PlayerState
    num_rows: int
    num_cols: int
    directions: List[Tuple[int, int]]
    phase_to_phase_factor: Dict[int, float]

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.search_algo = AlphaBeta(utils.heuristic_function, utils.successor_states_of, utils.perform_move)
        self.my_state_list = []
        self.game_time_left = game_time
        self.phase_to_phase_factor = {1: 1/self.SQRT_2, 3: 1/self.SQRT_2, 2: self.SQRT_2}

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
        # starting external timer to update the remaining game_time_left correctly
        make_move_start_time = time.time()
        tick = make_move_start_time

        # setting up variables for the anytime-minimax
        phase = None
        should_continue_to_next_iteration = True
        current_depth = 0
        last_minimax_value = 0
        last_best_move = (0, 0)

        num_blocks_on_board = len(np.where(self.current_state.board == BLOCK_CELL))
        num_zeros_on_board = self.num_rows * self.num_cols - num_blocks_on_board

        # updating scores
        self.current_state.my_score, self.current_state.rival_score = players_score
        self.current_state.turn = MY_TURN

        smart_time_limit = self.game_time_left / np.ceil(num_zeros_on_board / 2)

        # if game_time left is very low, calculate time naively
        if self.game_time_left > 0.1 * self.game_time:
            phase = np.ceil(3 * num_blocks_on_board / (self.num_rows * self.num_cols))
            assert phase in [1, 2, 3]

            phase_factor = self.phase_to_phase_factor[phase]
            smart_time_limit *= phase_factor

        final_time_limit = min(time_limit, smart_time_limit)
        time_left = final_time_limit

        while should_continue_to_next_iteration:
            current_depth += 1
            last_minimax_value, last_best_move = self.search_algo.search(self.current_state, current_depth, True)

            # time management
            tock = time.time()
            time_diff = tock - tick
            time_left -= time_diff
            tick = time.time()

            is_there_no_time_for_next_iteration = time_left < BRANCHING_FACTOR * time_diff or \
                                                  time_left <= 0.1 * final_time_limit

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
        if phase is None:
            phase = 'no phase computed'
        print('global alpha-beta:')
        print(f'turn: {len(self.my_state_list)}, depth: {current_depth}, minimax value: {last_minimax_value} '
              f'scores: player1: {self.current_state.my_score} ,player2: {self.current_state.rival_score}, '
              f'phase: {phase}')

        make_move_end_time = time.time()

        self.game_time_left -= make_move_end_time - make_move_start_time

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
