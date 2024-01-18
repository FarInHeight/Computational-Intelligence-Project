from game import Game, Move, Player
from copy import deepcopy
from itertools import product
from typing import Literal, Any
from utils.symmetry import Symmetry
import numpy as np
from collections import defaultdict

POSSIBLE_MOVES = []
# for each piece position
for from_pos in set(product([0, 4], range(Game()._board.shape[0]))).union(
    set(product(range(Game()._board.shape[1]), [0, 4]))
):
    # create a list of possible slides
    slides = list(Move)
    # for each slide
    for slide in slides:
        # make a copy of the current game state
        state = deepcopy(Game())
        # create an action
        action = (from_pos, slide)
        # perfom the action (note: _Game__move is necessary due to name mangling)
        ok = state._Game__move(from_pos, slide, 0)
        # if it is a valid action
        if ok:
            # append to the list of possible moves
            POSSIBLE_MOVES.append(action)


class MissNoAddDict(defaultdict):
    """
    Class extending defaultdict to not add a new key if it is not present.
    """

    def __missing__(self, __key: Any) -> Any:
        """
        If the key is missing, don't create a new dictionary entry
        and return the default value.

        Args:
            __key: the key used to index the dictionary.

        Returns:
            The default factory value is returned.
        """
        return self.default_factory()


class InvestigateGame(Game):
    '''
    Class representing an extension of the Game class which
    is used by the players to simulate the game or to train them.
    '''

    def __init__(self, game: 'Game') -> None:
        '''
        Constructor for creating a copy of the game to investigate.

        Args:
            game: the current game state.

        Returns:
            None.
        '''
        super().__init__()
        self._board = game.get_board()
        self.current_player_idx = game.get_current_player()
        self._player_to_symbol = {-1: '⬜️', 0: '❌', 1: '⭕️'}

    def print(self) -> None:
        '''
        Print the board in a fancy way.

        Args:
            None.

        Returns:
            None
        '''

        # for each row
        for i in range(self._board.shape[0]):
            # for each column
            for j in range(self._board.shape[1]):
                # print the element in position (i, j)
                print(self._player_to_symbol[self._board[i, j]], end='')
            # go to the beginning of the next line
            print()

    def __eq__(self, other: 'InvestigateGame') -> bool:
        '''
        Equality check for the class. Games are equal if the boards are equal.

        Args:
            other: another game instance.

        Returns:
            The equality of two games is returned.
        '''
        return (self._board == other._board).all()

    def get_hashable_state(self, player_id: int) -> int:
        '''
        Get a unique representation of a state that can be used as a key for a dictionary.

        Args:
           player_id: the player's id.

        Returns:
            An integer representation of the state is returned.
        '''
        # map the game state to an integer in base 3
        return int(''.join(str(_) for _ in (self._board + 1).flatten()) + str(player_id), base=3)

    def generate_possible_transitions(self) -> list[tuple[tuple[tuple[int, int], Move], 'InvestigateGame']]:
        '''
        Generate all possible game transitions that the current player can make.

        Args:
            None.

        Returns:
            A list of 3-length tuples of actions, corresponding game states and their
            representations is returned.
        '''
        # define a list of possible transitions
        transitions = []
        # for each piece position
        for from_pos, slide in POSSIBLE_MOVES:
            # make a copy of the current game state
            state = deepcopy(self)
            # create an action
            action = (from_pos, slide)
            # perfom the action (note: _Game__move is necessary due to name mangling)
            ok = state._Game__move(from_pos, slide, self.current_player_idx)
            # if it is a valid action
            if ok:
                # update the current player index
                state.current_player_idx = 1 - state.current_player_idx
                # append to the list of possible transitions
                transitions.append((action, state, state.get_hashable_state(self.current_player_idx)))

        return transitions

    def generate_canonical_transitions(self) -> list[tuple[tuple[tuple[int, int], Move], 'InvestigateGame']]:
        '''
        Generate all possible canonical game transitions that the current player can make.
        All transitions that have the same canonical representation of the resulting game
        state are not returned.

        Args:
            None.

        Returns:
            A list of 3-length tuples of actions, corresponding game states and their
            canonical representations is returned.
        '''
        # define a list of possible transitions
        transitions = []
        # define a set of canonical states
        canonical_states = set()
        # for each piece position
        for from_pos, slide in POSSIBLE_MOVES:
            # make a copy of the current game state
            state = deepcopy(self)
            # create an action
            action = (from_pos, slide)
            # perfom the action (note: _Game__move is necessary due to name mangling)
            ok = state._Game__move(from_pos, slide, self.current_player_idx)
            # if it is a valid action
            if ok:
                # get the equivalent canonical state
                canonical_state = Symmetry.get_canonical_state(state, self.current_player_idx)
                # if it is a new canonical state
                if canonical_state not in canonical_states:
                    # update the current player index
                    state.current_player_idx = 1 - state.current_player_idx
                    # append to the list of possible transitions
                    transitions.append((action, state, canonical_state))
                    # appens to the list of canonical states
                    canonical_states.add(canonical_state)

        return transitions

    def evaluation_function(self, player_id: int, enhance: bool = False) -> int | float:
        """
        Given the current state of the game, a static evaluation is performed
        and it is determined if the current position is an advantageous one
        for the Chosen Player or the Opponent player.
        Values greater than zero indicate that Chosen Player will probably win,
        zero indicates a balanced situation and values lower than
        zero indicate that Opponent Player will probably win.
        The value is calculated as:
        (number of complete rows, columns, or diagonals that are still open
        for Chosen Player) - (number of complete rows, columns, or diagonals that are
        still open for Opponent Player)

        Args:
            game: the current game state;
            player_id: the player's id.
            enhance: choose whether to weight a row according to the number of items taken.

        Return:
            An estimate value of how much a Chosen Player is winning
            or losing is returned.
        """

        # check if the game is over
        value = self.check_winner()
        # take the current player id
        current_player = player_id
        # take the opponent player id
        opponent_player = 1 - player_id

        # if the current player wins return +inf
        if value == current_player:
            return float('inf')
        # if opponent player wins return -inf
        elif value == opponent_player:
            return float('-inf')

        # turn each player id into a type compatible with the game board
        current_player = np.int16(current_player)
        opponent_player = np.int16(opponent_player)

        # define the current player's value
        current_player_value = 0
        # define the opponent player's value
        opponent_player_value = 0
        # get the board
        board = self.get_board()

        # define which board lines to examine
        lines = (
            # take all rows
            [board[y, :] for y in range(board.shape[0])]
            # take all columns
            + [board[:, x] for x in range(board.shape[1])]
            # take the principal diagonal
            + [[board[y, y] for y in range(board.shape[0])]]
            # take the secondary diagonal
            + [[board[y, -(y + 1)] for y in range(board.shape[0])]]
        )

        # for each board line
        for line in lines:
            # count the taken pieces by the current player
            current_player_taken = (line == current_player).sum()
            # count the taken pieces by the opponent player
            opponent_player_taken = (line == opponent_player).sum()
            # if all the pieces are neutral or belong to the current player
            if opponent_player_taken == 0 and current_player_taken > 0:
                # increment the current player's value
                current_player_value += current_player_taken if enhance else 1
            # if all the pieces are neutral or belong to the opponent player
            if current_player_taken == 0 and opponent_player_taken > 0:
                # increment the opponent player's value
                opponent_player_value += opponent_player_taken if enhance else 1

        return current_player_value - opponent_player_value

    def play(
        self,
        player1: 'Player',
        player2: 'Player',
        max_steps_draw: int,
    ) -> Literal[-1, 0, 1]:
        '''
        Play a game between two given players.

        Args:
            player1: the player who starts the game;
            player2: the second player of the game;
            max_steps_draw: define the maximum number of steps before
                            claiming a draw (avoid looping).

        Returns:
            0 is returned if the first player has won;
            1 is returned if the second player has won;
            -1 is returned if no one has won.
        '''
        # print beginning of the game
        print('-- BEGINNING OF THE GAME --')
        # print the board
        self.print()
        # define the players
        players = [player1, player2]
        # set the moving player index
        self.current_player_idx = 1
        # define a variable to indicate if there is a winner
        winner = -1
        # define counter to terminate if we are in a loop
        counter = 0
        # save last played actions
        last_actions = [None, None]
        # if we can still play
        while winner < 0 and counter < max_steps_draw:
            # update the current moving player index
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            # define a variable to check if the chosen move is ok or not
            ok = False
            # while the chosen move is not ok
            while not ok:
                # let the current player make a move
                from_pos, slide = players[self.current_player_idx].make_move(self)
                # check if now it is ok
                ok = self._Game__move(from_pos, slide, self.current_player_idx)
            # if we play the same action as before
            if last_actions[self.current_player_idx] == (from_pos, slide):
                # increment the counter
                counter += 0.5
            # otherwise
            else:
                # save the new last action
                last_actions[self.current_player_idx] = (from_pos, slide)
                # reset the counter
                counter = 0
            # print the move
            print(
                f'Player {self._player_to_symbol[self.current_player_idx]} chose to move {from_pos} to the {slide.name.lower()}'
            )
            # print the board
            self.print()
            # check if there is a winner
            winner = self.check_winner()
        # print who won
        if winner == -1:
            print(f"Early Stopping: Draw!")
        else:
            print(f"Winner: Player {winner}")
        # print the end of the game
        print('-- END OF THE GAME --')
        # return the winner
        return winner
