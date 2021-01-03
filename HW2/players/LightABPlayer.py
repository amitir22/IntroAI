"""
MiniMax Player with AlphaBeta pruning with light heuristic
"""
from players.AbstractPlayer import AbstractPlayer
from typing import Tuple, List
from SearchAlgos import AlphaBeta
import numpy as np
import utils
from utils import PlayerState, MY_TURN, RIVAL_TURN


class Player(AbstractPlayer):
    BRANCHING_FACTOR = 4
    BLOCK_CELL = -1
    P1_CELL = 1
    P2_CELL = 2

    game_time: float
    penalty_score: int
    search_algo: AlphaBeta
    current_state: PlayerState
    num_rows: int
    num_cols: int
    directions: List[Tuple[int, int]]
    depth: int

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.search_algo = AlphaBeta(Player.light_heuristic_function, utils.successor_states_of, utils.perform_move)
        self.my_state_list = []
        self.penalty_score = penalty_score
        self.game_time = game_time

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

        my_loc = np.where(board == self.P1_CELL)
        my_loc = my_loc[0][0], my_loc[1][0]
        rival_loc = np.where(board == self.P2_CELL)
        rival_loc = rival_loc[0][0], rival_loc[1][0]

        assert None not in [my_loc, rival_loc]

        board[my_loc] = Player.BLOCK_CELL
        board[rival_loc] = Player.BLOCK_CELL

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
        # updating scores
        self.current_state.my_score, self.current_state.rival_score = players_score
        self.current_state.turn = MY_TURN

        current_depth = self.depth

        if current_depth is None:
            raise NotImplementedError('LightABPlayer data-member "depth" must be set before running the game.')

        minimax_value, best_move = self.search_algo.search(self.current_state, current_depth, True)

        # updating next state and returning the next move
        self.current_state.fruits_turns_to_live -= 1
        next_location = utils.tup_add(self.current_state.my_loc, best_move)

        assert best_move != (0, 0)

        # updating the board in the next state and moving the current_state to point at the next
        self.current_state = utils.perform_move(self.current_state, MY_TURN, next_location)
        self.my_state_list.append((self.current_state, minimax_value, best_move))

        # todo: remove
        print('light:')
        print(f'turn: {len(self.my_state_list)}, depth: {current_depth}, '
              f'scores: player1: {self.current_state.my_score} ,player2: {self.current_state.rival_score}')

        return best_move

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

    ########## helper functions for MiniMax algorithm ##########

    # TODO: make sure this is ok
    @staticmethod
    def light_heuristic_function(state: PlayerState):
        is_not_hole_state = False

        for direction in utils.get_directions():
            new_loc = utils.tup_add(state.my_loc, direction)

            if utils.is_move_valid(state.board, new_loc):
                is_not_hole_state = True
                break

        score = state.my_score - state.rival_score

        if is_not_hole_state:
            return score
        else:
            return score - state.penalty_score
