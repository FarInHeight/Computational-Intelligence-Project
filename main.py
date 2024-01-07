from game import Game
from min_max import MinMaxPlayer, AlphaBetaMinMaxPlayer
from random_player import RandomPlayer
from human_player import HumanPlayer
from q_learning import QLearningRLPlayer
import time

if __name__ == '__main__':
    g = Game()
    g.print()
    player1 = AlphaBetaMinMaxPlayer(0, depth=1)
    # player1 = RandomPlayer()
    # player2 = RandomPlayer()
    # player2 = AlphaBetaMinMaxPlayer(1, depth=5, symmetries=True)
    player2 = q_learning_rl_agent = QLearningRLPlayer(
        n_episodes=10_000,
        alpha=0.1,
        gamma=0.99,
        min_exploration_rate=0.01,
        exploration_decay_rate=1e-4,
        minmax=True,
    )
    player2.load('agents/q_learning_rl_agent.pkl')
    start = time.time()
    winner = g.play(player1, player2)
    total_time = time.time() - start
    g.print()
    print(f"Winner: Player {winner}")
    print(f'Game duration: {total_time:.2E} sec, {total_time / 60:.2E} min')
