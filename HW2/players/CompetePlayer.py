"""
Player for the competition
"""
from typing import Tuple, List, Dict
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import AlphaBeta
import time
import numpy as np
import networkx as nx
import utils
from utils import PlayerState, BRANCHING_FACTOR, BLOCK_CELL, P1_CELL, P2_CELL, MY_TURN, RIVAL_TURN
# TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    NAME = 'Agent-SARS-CoV-007'
    SQRT_2 = 2.0 ** 0.5

    game_time_left: float
    game_time: float
    penalty_score: int
    search_algo: AlphaBeta
    current_state: PlayerState
    num_rows: int
    num_cols: int
    directions: List[Tuple[int, int]]
    phase_to_phase_factor: Dict[int, float]

    # TODO: remove before submission. here for debugging purposes.
    my_state_list: List[Tuple[PlayerState, int, Tuple[int, int]]]

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.search_algo = AlphaBeta(Player.heuristic_function, Player.successor_states_of,
                                     Player.perform_move)
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
                                         player=self, turn=MY_TURN)

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
        tick = time.time()

        # setting up variables for the anytime-minimax
        phase = None  # todo: remove
        should_continue_to_next_iteration = True
        current_depth = 0
        last_minimax_value = 0
        last_best_move = (0, 0)

        num_blocks_on_board = len(np.where(self.current_state.board == BLOCK_CELL)[0])
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

            # todo: remove
            assert last_best_move not in [(0, 0), None]

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
        self.current_state = Player.perform_move(self.current_state, MY_TURN, next_location)
        self.my_state_list.append((self.current_state, last_minimax_value, last_best_move))

        # todo: remove
        if phase is None:
            phase = 'no phase computed'
        print('contest player:')
        print(f'turn: {len(self.my_state_list)}, depth: {current_depth}, '
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
        self.current_state = Player.perform_move(self.current_state, RIVAL_TURN, pos)

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
            is_not_blocked = board[target_location] != BLOCK_CELL

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

    ########## helper functions for the search algorithm ##########

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
            next_state.board[target_location] = BLOCK_CELL

            return next_state
        return None

    @staticmethod
    def successor_states_of(state: PlayerState):
        """
        :param state: the state we expand
        :return: tuple (a successor state, the direction to the successor)
        """
        heuristic_values_of_state = []
        successor_states = []

        for direction in state.player.directions:
            player_old_loc = state.players_locations[state.turn]
            player_new_loc = utils.tup_add(player_old_loc, direction)

            if Player.is_move_valid(state.board, player_new_loc):
                next_state = Player.perform_move(state, state.turn, player_new_loc)

                next_state.fruits_turns_to_live -= 1
                next_state.turn = 1 - state.turn  # if rival -> next=me | else next=rival

                h_value = Player.heuristic_function(next_state)

                heuristic_values_of_state.append(h_value)
                successor_states.append((next_state, direction))

        sorted_indices = np.argsort(heuristic_values_of_state)
        returned_successor_states = []

        for index in sorted_indices:
            returned_successor_states.append(successor_states[index])

        return returned_successor_states

    @staticmethod
    def heuristic_function(state: PlayerState):
        a = 2  # todo: change

        my_row, my_col = state.players_locations[state.turn]

        score_available_cells = 1
        score_blocked_cells = 1

        for row in range(my_row - a, my_row + a + 1):
            for col in range(my_col - a, my_col + a + 1):
                if (row, col) != (my_row, my_col):
                    if Player.is_location_in_board(state.board, (row, col)):
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

            my_fruit_dist = Player.m_dist(state.my_loc, fruit_loc)
            rival_fruit_dist = Player.m_dist(state.rival_loc, fruit_loc)

            is_fruit_viable = my_fruit_dist <= rival_fruit_dist and my_fruit_dist < state.fruits_turns_to_live

            if is_fruit_viable and (my_fruit_dist, fruit_score) < (current_min_fruit_dist, current_min_fruit_score):
                current_min_fruit_dist, current_min_fruit_score = my_fruit_dist, fruit_score

        score_closest_fruit_m_dist = current_min_fruit_dist
        score_adversary_m_dist = Player.m_dist(state.my_loc, state.rival_loc) + \
                                 np.linalg.norm(np.array(state.my_loc) - np.array(state.rival_loc))

        possible_loc1 = my_row, my_col + 1
        possible_loc2 = my_row, my_col - 1
        possible_loc3 = my_row + 1, my_col
        possible_loc4 = my_row - 1, my_col

        is_not_hole_state = int(Player.is_move_valid(state.board, possible_loc1) or
                                Player.is_move_valid(state.board, possible_loc2) or
                                Player.is_move_valid(state.board, possible_loc3) or
                                Player.is_move_valid(state.board, possible_loc4))

        # todo: check if need to add '+1' in previous players
        return is_not_hole_state * ((score_available_cells / score_blocked_cells) /
                                    (score_closest_fruit_m_dist + score_adversary_m_dist + 1))

    @staticmethod
    def asdf(state: PlayerState):
        board = state.board
        g = nx.Graph()

        num_rows = len(board)
        num_cols = len(board[0])

        g.add_nodes_from([(row, col) for row in range(num_rows) for col in range(num_cols)])

        for row in range(num_rows):
            for col in range(num_cols):
                loc = row, col

                to_the_right = row, col + 1
                upwards = row + 1, col

                if col + 1 < num_cols:
                    g.add_edge(loc, to_the_right)
                if row + 1 < num_rows:
                    g.add_edge(loc, upwards)

        for row in range(num_rows):
            for col in range(num_cols):
                loc = row, col
                if board[loc] in [BLOCK_CELL, P2_CELL]:
                    g.remove_node(loc)

        my_loc_component = Player.get_component_contains_location(g, state.my_loc)
        rival_loc_component = Player.get_component_contains_location(g, state.rival_loc)

        my_target_vertex = np.argmax(Player.m_dist_component_nodes(my_loc_component, state.my_loc))
        rival_target_vertex = np.argmax(Player.m_dist_component_nodes(rival_loc_component, state.rival_loc))

        # todo: not good enough, because maybe i don't need to set my target so far away... maybe consider DFS-post-ord.
        my_longest_path = Player.get_longest_path(state.my_loc, my_loc_component, my_target_vertex)
        rival_longest_path = Player.get_longest_path(state.rival_loc, rival_loc_component, rival_target_vertex)

        # todo: divide to components
        # todo: locate component of my_loc
        # todo: suspicious for removal: vertices in my_loc component which aren't my_loc and is of rank = 1
        # if there are more than 1, let's say... 3? then first try to remove one of them at a time and check if
        # eulerian, if found then remove and return eulerian path length, if still fails try removing 2 at a time and
        # so on...

        return my_longest_path, rival_longest_path

    @staticmethod
    def get_longest_path(loc: Tuple[int, int], loc_component: nx.Graph, target_vertex_index):
        target_vertex = loc_component.nodes[target_vertex_index]

        return max((path for path in nx.all_simple_paths(loc_component, loc, target_vertex)))

    @staticmethod
    def m_dist_component_nodes(my_loc_component: nx.Graph, src: Tuple[int, int]):
        return [Player.m_dist(src, loc) for node, loc in my_loc_component.nodes(data=True)]

    @staticmethod
    def get_component_contains_location(g: nx.Graph, loc: Tuple[int, int]):
        return g.subgraph(set(nx.algorithms.components.node_connected_component(g, loc))).copy()


# TODO: delete
def main():
    board = np.array([[1, -1,  2],
                      [0,  0,  0],
                      [0,  0,  0]])
    my_loc = np.where(board == P1_CELL)
    my_loc = my_loc[0][0], my_loc[1][0]
    rival_loc = np.where(board == P2_CELL)
    rival_loc = rival_loc[0][0], rival_loc[1][0]

    state = PlayerState(board=board, fruit_locations={}, fruits_turns_to_live=0,
                        players_scores={}, player=AbstractPlayer(0.0, 0), turn=MY_TURN,
                        players_locations={MY_TURN: my_loc, RIVAL_TURN: rival_loc})
    path_length = Player.asdf(state)
    print(path_length)


# TODO: delete
if __name__ == '__main__':
    main()
