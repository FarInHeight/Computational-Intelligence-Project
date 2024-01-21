from game import Game, Move, Player


class HumanPlayer(Player):
    """
    Class that allows a human being to play.
    """

    def __init__(self) -> None:
        """
        Constructor of the human player.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__()
        self.__slides = {"r": Move.RIGHT, "l": Move.LEFT, "t": Move.TOP, "b": Move.BOTTOM}

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Play a valid move.

        Args:
            game: a game instance.

        Returns:
            A move to play is returned.
        """
        # set the move as invalid
        valid = False
        # while the chosen move is not valid
        while not valid:
            # let the user choose the move
            valid, action = self.__get_input(game)
        # return the chosen move
        return action

    def __get_input(self, game: 'Game') -> tuple[bool, tuple[tuple[int, int], Move] | None]:
        """
        Ask the user to enter a move.

        Args:
            game: a game instance.

        Return:
            A boolean value to determine whether an input is OK and t
            he input itself is returned.
        """
        # get the current board
        board = game.get_board()
        # set the move as invalid
        valid, action = False, None
        try:
            # let the user decide the x-value
            x = int(input('Choose a position in the x-axis (→) in [0-4]: '))
            # if the x-value is not valid
            if x < 0 or x >= board.shape[1]:
                # print a descriptive message
                print('Invalid integer value. Please insert an integer between 0 and 4.')
                # return an invalid move
                valid, action = False, None
            else:
                # let the user decide the y-value
                y = int(input('Choose a position in the y-axis (↓) in [0-4]: '))
                # if the y-value is not valid
                if y < 0 or y >= board.shape[0]:
                    # print a descriptive message
                    print('Invalid integer value. Please insert an integer between 0 and 4.')
                    # return an invalid move
                    valid, action = False, None
                # otherwise
                else:
                    # let the user decide a slide
                    slide = self.__slides[input('Choose a slide in {"r", "l", "t", "b"}: ')]
                    # save the chosen move
                    valid, action = True, ((x, y), slide)
        # if the user chooses a non-integer value for x or y
        except ValueError:
            # print a descriptive message
            print('Invalid non-integer value.')
        # if the user chooses an invalid slide
        except KeyError:
            # print a descriptive message
            print('Invalid slide value.')
        # at the end
        finally:
            # return the chosen move or the invalid move
            return valid, action


if __name__ == '__main__':
    # test make_move method
    g = Game()
    player = HumanPlayer()
    print(player.make_move(g))
