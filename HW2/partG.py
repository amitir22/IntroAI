import argparse
import players.HeavyABPlayer
import players.LightABPlayer
import utils
from GameWrapper import GameWrapper


def ex18():
    pass


def ex19():
    parser = argparse.ArgumentParser()

    parser.add_argument('-heavy_depth', type=int)
    parser.add_argument('-light_depth', type=int)
    parser.add_argument('-board', default='default_board.csv', type=str,
                        help='Name of board file (.csv).')

    args = parser.parse_args()

    heavy_player_depth = args.heavy_depth
    light_player_depth = args.light_depth
    board_name = args.board

    game_time = 1000
    move_time = 1000
    penalty_score = 300

    board_dims, blocks, players_locations = utils.get_board_from_csv(board_name)

    player1 = players.HeavyABPlayer.Player(game_time=game_time, penalty_score=penalty_score)
    player2 = players.LightABPlayer.Player(game_time=game_time, penalty_score=penalty_score)
    player1.depth = heavy_player_depth
    player2.depth = light_player_depth

    game_wrapper = GameWrapper(size=board_dims, block_positions=blocks, starts=players_locations,
                               player_1=player1, player_2=player2, terminal_viz=False, print_game_in_terminal=False,
                               time_to_make_a_move=move_time, game_time=game_time, penalty_score=penalty_score)
    game_wrapper.start_game()


def main():
    ex18()
    ex19()


if __name__ == '__main__':
    main()
