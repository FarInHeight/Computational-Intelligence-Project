from game import Game
from utils.investigate_game import InvestigateGame
from players.random_player import RandomPlayer


if __name__ == '__main__':
    g = Game()
    g.play(RandomPlayer(), RandomPlayer())
