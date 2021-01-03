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
    args = parser.parse_args()

    heavy_player_depth = args.heavy_depth
    light_player_depth = args.light_depth

    boards_names = ['default_board.csv', 'custom_board.csv'] + \
                   ['custom_board' + str(i) + '.csv' for i in range(2, 5 + 1)]
    game_time = 1000
    move_time = 1000
    penalty_score = 300
    number_of_games_foreach_board = 5
    board_index = 0
    boards_name_to_players_win_counters = {}

    for board_name in boards_names:
        player1_win_counter = 0
        player2_win_counter = 0

        for game_index in range(number_of_games_foreach_board):
            try:
                board_dims, blocks, players_locations = utils.get_board_from_csv(board_name)
                player1_loc, player2_loc = tuple(players_locations)

                player1 = players.HeavyABPlayer.Player(game_time=game_time, penalty_score=penalty_score)
                player2 = players.LightABPlayer.Player(game_time=game_time, penalty_score=penalty_score)
                player1.depth = heavy_player_depth
                player2.depth = light_player_depth

                game_wrapper = GameWrapper(size=board_dims, block_positions=blocks, starts=players_locations,
                                       player_1=player1, player_2=player2, terminal_viz=False, print_game_in_terminal=False,
                                       time_to_make_a_move=move_time, game_time=game_time, penalty_score=penalty_score)
                game_wrapper.start_game()
            except RuntimeError:
                game = game_wrapper.game
                player1_score, player2_score = tuple(game.get_players_scores())

                if player1_score > player2_score:
                    player1_win_counter += 1
                elif player1_score < player2_score:
                    player2_win_counter += 1
                # else - do nothing todo: maybe rerun?

                board_index += 1

        print(player1_win_counter, player2_win_counter)


def main():
    ex18()
    ex19()


if __name__ == '__main__':
    main()
