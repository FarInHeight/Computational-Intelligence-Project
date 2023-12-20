from game import Game
from min_max import MinMaxPlayer, AlphaBetaMinMaxPlayer
from random_player import RandomPlayer
from human_player import HumanPlayer


if __name__ == '__main__':
    g = Game()
    g.print()
    # player1 = AlphaBetaMinMaxPlayer(0, depth=4)
    player1 = RandomPlayer()
    # player2 = RandomPlayer()
    player2 = AlphaBetaMinMaxPlayer(1, depth=4)
    winner = g.play(player1, player2)
    g.print()
    print(f"Winner: Player {winner}")
