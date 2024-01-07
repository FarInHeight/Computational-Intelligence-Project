from game import Game, Move
from copy import deepcopy
from itertools import product
import numpy as np


class InvestigateGame(Game):
    """
    Class representing an extension of the Game class which
    is used by the Min-Man players to construct the search tree.
    """

    def __init__(self, game: 'Game') -> None:
        """
        Constructor for creating a copy of the game to investigate.

        Args:
            game: the current game state.

        Returns:
            None.
        """
        super().__init__()
        self._board = game.get_board()
        self._rotations = [lambda x: np.rot90(x, k=1), lambda x: np.rot90(x, k=2), lambda x: np.rot90(x, k=3)]
        self._flips = [lambda x: np.fliplr(x), lambda x: np.flipud(x)]

    def __eq__(self, other) -> bool:
        '''
        Equality check for the class. Games are equal if the boards are equal.

        Args:
            other: another game instance.

        Returns:
            The equality of two games is returned.
        '''
        return (self._board == other._board).all()

    def _get_transformed_states(self) -> list['InvestigateGame']:
        '''
        Apply all possible transformations to the state and return a list of equivalent transformed states.
        To compute all equivalent states, apply a 90°, 180°, and 270° rotation.
        A horizontal and vertical flip, and a 90° rotation on the two flipped states.
        Because a 180° rotation of the flipped states is equivalent to a flip in the original position,
        and a 270° rotation of the flipped states is equivalent to flipping the 90° rotation.
        (270° rotation of horizontal/vertical flip = 90° rotation of vertical/horizontal flip) and
        (horizontal/vertical flip = 180° rotation of vertical/horizontal flip).

        Args:
            None.

        Returns:
            A list of equivalent transformed states is returned.
        '''

        # define list of transformed states
        transformed_states = [self]

        # add flipped and flipped + rotate90 states
        for flip in self._flips:
            # copy of the state
            state = deepcopy(self)
            # transform the board
            flipped_board = flip(state._board)
            # new flipped state
            state._board = flipped_board
            # append transformed state
            transformed_states.append(state)
            # copy of the state
            state = deepcopy(self)
            # new state flipped and rotated
            state._board = np.rot90(flipped_board)
            # append new transformed state
            transformed_states.append(state)

        # add rotated states
        for rotate in self._rotations:
            # copy of the state
            state = deepcopy(self)
            # transform the board
            state._board = rotate(state._board)
            # append transformed state
            transformed_states.append(state)

        return transformed_states

    def generate_possible_transitions(
        self, player_id: int
    ) -> list[tuple[tuple[tuple[int, int], Move], 'InvestigateGame']]:
        """
        Generate all possible game transitions that a given player can make.

        Args:
            player_id: the player's id.

        Returns:
            A list of pairs of actions and corresponding game states
            is returned.
        """
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
        """
        Generate all possible game transitions that a given player can make considering the canonical states.

        Args:
            player_id: the player's id.

        Returns:
            A list of pairs of actions and corresponding game states
            is returned.
        """

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
            transformed_states = states[i]._get_transformed_states()
            # delete transformed states
            states = states[: i + 1] + [state for state in states[i + 1 :] if state not in transformed_states]
            # increment index
            i += 1

        return [transitions for transitions in transitions if transitions[1] in states]
