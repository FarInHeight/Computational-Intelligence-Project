from random import randint, choice
from game import Game, Move, Player


class RandomPlayer(Player):
    """
    Class representing a player which plays randomly.
    """

    def __init__(self) -> None:
        """
        Constructor of the Random player.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Select a random move to play.

        Args:
            game: the current game state.

        Return:
            A random move is returned.
        """
        board = game.get_board()
        from_pos = (randint(0, board.shape[1] - 1), randint(0, board.shape[0] - 1))
        move = choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
