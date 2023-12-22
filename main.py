from game import Game
from min_max import MinMaxPlayer, AlphaBetaMinMaxPlayer
from random_player import RandomPlayer
from human_player import HumanPlayer
import time

if __name__ == '__main__':
    g = Game()
    g.print()
    # player1 = AlphaBetaMinMaxPlayer(0, depth=4)
    player1 = RandomPlayer()
    # player2 = RandomPlayer()
    player2 = AlphaBetaMinMaxPlayer(1, depth=4, symmetries=True)
    start = time.time()
    winner = g.play(player1, player2)
    total_time = time.time() - start
    g.print()
    print(f"Winner: Player {winner}")
    print(f'Game duration: {total_time:.2f} sec, {total_time / 60:.2f} min')
