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
    player2 = AlphaBetaMinMaxPlayer(1, depth=3, symmetries=True)
    start = time.time()
    winner = g.play(player1, player2)
    total_time = time.time() - start
    g.print()
    print(f"Winner: Player {winner}")
    print(f'Game duration: {total_time} sec, {total_time/60} min')

    """ 
    # test generate_canonical_transitions() execution time vsgenerate_possible_transitions()
    start = time.time()
    transictions = AlphaBetaMinMaxPlayer(1).InvestigateGame(g).generate_canonical_transitions(1)
    total_time = time.time() - start
    print(f'Can: {total_time} sec, {total_time/60} min')

    start = time.time()
    transictions = AlphaBetaMinMaxPlayer(1).InvestigateGame(g).generate_possible_transitions(1)
    total_time = time.time() - start
    print(f'Non can: {total_time} sec, {total_time/60} min') """
