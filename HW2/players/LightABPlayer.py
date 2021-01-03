"""
MiniMax Player with AlphaBeta pruning with light heuristic
"""
from players.AbstractPlayer import AbstractPlayer
from typing import Tuple, List, Dict
from SearchAlgos import AlphaBeta
import numpy as np
import dataclasses
import copy
import utils
#TODO: you can import more modules, if needed


@dataclasses.dataclass(init=True)
class PlayerState:
    MY_TURN = 0
    RIVAL_TURN = 1

    turn: int
    board: np.array  # matrix
    fruit_locations: Dict[Tuple[int, int], int]  # key: location, value: fruit_score
    fruits_turns_to_live: int
    players_locations: Dict[int, Tuple[int, int]]
    players_scores: Dict[int, int]
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
    search_algo: AlphaBeta
    current_state: PlayerState
    num_rows: int
    num_cols: int
    directions: List[Tuple[int, int]]
    depth: int

    # TODO: remove before submission. here for debugging purposes.
    my_state_list: List[Tuple[PlayerState, int, Tuple[int, int]]]

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.search_algo = AlphaBeta(Player.heuristic_function, Player.successor_states_of,
                                     Player.perform_move)
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

        players_scores = {PlayerState.MY_TURN: 0, PlayerState.RIVAL_TURN: 0}
        players_locations = {PlayerState.MY_TURN: my_loc, PlayerState.RIVAL_TURN: rival_loc}

        fruits_turns_to_live = min(self.num_rows, self.num_cols)

        self.current_state = PlayerState(board=board, fruit_locations={}, fruits_turns_to_live=fruits_turns_to_live,
                                         players_scores=players_scores, players_locations=players_locations,
                                         player=self, turn=PlayerState.MY_TURN)

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
        self.current_state.turn = PlayerState.MY_TURN

        current_depth = self.depth

        if current_depth is None:
            raise NotImplementedError('LightABPlayer data-member "depth" must be set before running the game.')

        minimax_value, best_move = self.search_algo.search(self.current_state, current_depth, True)

        # updating next state and returning the next move
        self.current_state.fruits_turns_to_live -= 1
        next_location = utils.tup_add(self.current_state.my_loc, best_move)

        assert best_move != (0, 0)

        # updating the board in the next state and moving the current_state to point at the next
        self.current_state = Player.perform_move(self.current_state, PlayerState.MY_TURN, next_location)
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
        self.current_state = Player.perform_move(self.current_state, PlayerState.RIVAL_TURN, pos)

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

    @staticmethod
    def is_move_valid(board: np.array, target_location: Tuple[int, int]):
        if Player.is_location_in_board(board, target_location):
            is_not_blocked = board[target_location] != Player.BLOCK_CELL

            return is_not_blocked
        else:
            return False

    @staticmethod
    def is_location_in_board(board: np.array, location: Tuple[int, int]):
        num_rows = len(board)
        num_cols = len(board[0])

        row, col = location

        return 0 <= row < num_rows and 0 <= col < num_cols

    @staticmethod
    def m_dist(loc1: Tuple[int, int], loc2: Tuple[int, int]):
        row1, col1 = loc1
        row2, col2 = loc2

        return np.abs(row2 - row1) + np.abs(col2 - col1)

    ########## helper functions for MiniMax algorithm ##########

    @staticmethod
    def perform_move(state: PlayerState, player_turn: int, target_location: Tuple[int, int]):
        """
        input:
            :param state: the state which we update
            :param player_turn: if 0 -> moving my player, else (=1) -> moving rival player
            :param target_location: the target of the move
        output:
            :return: the updated state. None if the move given was not valid
        """
        next_state = state.duplicate()

        if Player.is_move_valid(next_state.board, target_location):
            if target_location in next_state.fruit_locations and state.fruits_turns_to_live > 0:
                del next_state.fruit_locations[target_location]
                next_state.players_scores[player_turn] += next_state.board[target_location]

            next_state.players_locations[player_turn] = target_location
            next_state.board[target_location] = Player.BLOCK_CELL

            return next_state
        return None

    @staticmethod
    def successor_states_of(state: PlayerState):
        """
        :param state: the state we expand
        :return: tuple (a successor state, the direction to the successor)
        """
        successor_states = []

        for direction in state.player.directions:
            player_old_loc = state.players_locations[state.turn]
            player_new_loc = utils.tup_add(player_old_loc, direction)

            if Player.is_move_valid(state.board, player_new_loc):
                next_state = Player.perform_move(state, state.turn, player_new_loc)

                next_state.fruits_turns_to_live -= 1
                next_state.turn = 1 - state.turn  # if rival -> next=me | else next=rival
                successor_states.append((next_state, direction))

        return successor_states

    @staticmethod
    def heuristic_function(state: PlayerState):
        is_not_hole_state = False

        for direction in state.player.directions:
            new_loc = utils.tup_add(state.my_loc, direction)

            if Player.is_move_valid(state.board, new_loc):
                is_not_hole_state = True
                break

        score = state.my_score - state.rival_score

        if is_not_hole_state:
            return score
        else:
            return score - state.player.penalty_score
