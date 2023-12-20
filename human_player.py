from game import Game, Move, Player


class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self._slides = {"r": Move.RIGHT, "l": Move.LEFT, "t": Move.TOP, "b": Move.BOTTOM}

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        valid = False
        while not valid:
            valid, action = self._get_input(game)
        return action

    def _get_input(self, game: 'Game'):
        baord = game.get_board()
        valid, action = False, None
        try:
            x = int(input('Choose a position in the x-axis (→) in [0-4]: '))
            if x < 0 or x >= baord.shape[1]:
                print('Invalid integer value. Please insert an integer between 0 and 4.')
                valid, action = False, None
            else:
                y = int(input('Choose a position in the y-axis (↓) in [0-4]: '))
                if y < 0 or y >= baord.shape[0]:
                    print('Invalid integer value. Please insert an integer between 0 and 4.')
                    valid, action = False, None
                else:
                    slide = self._slides[input('Choose a slide in {"r", "l", "t", "b"}: ')]
                    valid, action = True, ((x, y), slide)
        except ValueError:
            print('Invalid non-integer value.')
        except KeyError:
            print('Invalid slide value.')
        finally:
            return valid, action


if __name__ == '__main__':
    # test make_move method
    g = Game()
    player = HumanPlayer()
    print(player.make_move(g))
