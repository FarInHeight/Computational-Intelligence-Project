from game import Game, Move, Player
from copy import deepcopy
from itertools import product
from typing import Literal
from symmetry import Symmetry
import numpy as np


class InvestigateGame(Game):
    '''
    Class representing an extension of the Game class which
    is used by the Min-Man players to construct the search tree.
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

    def __eq__(self, other) -> bool:
        '''
        Equality check for the class. Games are equal if the boards are equal.

        Args:
            other: another game instance.

        Returns:
            The equality of two games is returned.
        '''
        return (self._board == other._board).all()

    def get_hashable_state(self, player_id: int) -> str:
        '''
        Get a unique representation of a state that can be used as a key for a dictionary.

        Args:
           player_id: the player's id.

        Returns:
            A string representation of the state is returned
        '''
        # copy of the state
        state = deepcopy(self)
        # change not taken tiles values to 0
        state._board += 1
        # map the trasformed_state to a string in base 3
        return ''.join(str(_) for _ in state._board.flatten()) + str(player_id)

    def generate_possible_transitions(
        self, player_id: int
    ) -> list[tuple[tuple[tuple[int, int], Move], 'InvestigateGame']]:
        '''
        Generate all possible game transitions that a given player can make.

        Args:
            player_id: the player's id.

        Returns:
            A list of pairs of actions and corresponding game states
            is returned.
        '''
        # define a list of possible transitions
        transitions = []
        # for each piece position
        for from_pos in product(range(self._board.shape[1]), range(self._board.shape[0])):
            # create a list of possible slides
            slides = list(Move)
            # shuffle(slides)
            # for each slide
            for slide in slides:
                # make a copy of the current game state
                state = deepcopy(self)
                action = (from_pos, slide)
                # perfom the move (note: _Game__move is necessary due to name mangling)
                ok = state._Game__move(from_pos, slide, player_id)
                # if it is valid
                if ok:
                    # append to the list of possible transitions
                    transitions.append((action, state))

        return transitions

    def generate_canonical_transitions(
        self, player_id: int
    ) -> list[tuple[tuple[tuple[int, int], Move], 'InvestigateGame']]:
        '''
        Generate all possible game transitions that a given player can make considering the canonical states.

        Args:
            player_id: the player's id.

        Returns:
            A list of pairs of actions and corresponding game states
            is returned.
        '''

        # get possible transitions
        transitions = self.generate_possible_transitions(player_id)
        # get the states
        _, states = zip(*transitions)
        # convert to list
        states = list(states)

        i = 0
        # for each transitions
        while i < len(states):
            # compute all the transformed states
            transformed_states = Symmetry.get_transformed_states(states[i])
            # delete transformed states
            states = states[: i + 1] + [state for state in states[i + 1 :] if state not in transformed_states]
            # increment index
            i += 1

        # create a list of unique transitions
        unique_transitions = []
        # for each transition
        for transition in transitions:
            # check that it refers to a unique state and that another
            # transition with the same final state has not already been added
            if transition[1] in states and not any(map(lambda x: x[1] == transition[1], unique_transitions)):
                # add another unique transition
                unique_transitions.append(transition)

        return unique_transitions

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
            log: a boolean flag to print the match log or not;
            max_steps_draw: define the maximum number of steps before
                            claiming a draw (avoid looping).

        Returns:
            0 is returned if the first player has won;
            1 is returned if the second player has won;
            -1 is returned if no one has won.
        '''
        # print beginning of the game
        print('-- BEGINNING OF THE GAME --')
        # print the boards
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
