from game import Move
import numpy as np
from copy import deepcopy


class Symmetry:
    '''
    Class that handles information about the symmetries of the board

    canonical_index_trasformation -> transformation
    0 : identity
    1 : rot90(  )
    2 : rot180(  )
    3 : rot270(  )
    4 : flipud(  )
    5 : rot90( flipud(  ) )
    6 : rot180( flipud(  ) )
    7 : rot270( flipud(  ) )

    '''

    def __init__(self) -> None:
        self._rotations = [lambda x: np.rot90(x, k=k) for k in range(1, 4)]
        self._flips = [lambda x: deepcopy(x), lambda x: np.flipud(x)]
        # mapping of canonical slide to original slide
        self._map_canonical_slide_to_original_slide = {
            1: {
                Move.BOTTOM: Move.LEFT,
                Move.LEFT: Move.TOP,
                Move.RIGHT: Move.BOTTOM,
                Move.TOP: Move.RIGHT,
            },
            2: {
                Move.LEFT: Move.RIGHT,
                Move.RIGHT: Move.LEFT,
                Move.BOTTOM: Move.TOP,
                Move.TOP: Move.BOTTOM,
            },
            3: {
                Move.LEFT: Move.BOTTOM,
                Move.RIGHT: Move.TOP,
                Move.BOTTOM: Move.RIGHT,
                Move.TOP: Move.LEFT,
            },
            4: {
                Move.LEFT: Move.LEFT,
                Move.RIGHT: Move.RIGHT,
                Move.BOTTOM: Move.TOP,
                Move.TOP: Move.BOTTOM,
            },
            5: {
                Move.LEFT: Move.BOTTOM,
                Move.RIGHT: Move.TOP,
                Move.BOTTOM: Move.LEFT,
                Move.TOP: Move.RIGHT,
            },
            6: {
                Move.LEFT: Move.RIGHT,
                Move.RIGHT: Move.LEFT,
                Move.BOTTOM: Move.BOTTOM,
                Move.TOP: Move.TOP,
            },
            7: {
                Move.BOTTOM: Move.RIGHT,
                Move.LEFT: Move.TOP,
                Move.RIGHT: Move.BOTTOM,
                Move.TOP: Move.LEFT,
            },
        }
        # canonical from position
        self._canonical_positions = np.array(
            [
                [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
                [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
                [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
                [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],
                [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
            ]
        )
        # mapping trasformation - trasformed from position
        self._trasformed_positions = {
            1: np.rot90(self._canonical_positions, axes=(0, 1)),
            2: np.rot90(self._canonical_positions, k=2, axes=(0, 1)),
            3: np.rot90(self._canonical_positions, k=3, axes=(0, 1)),
            4: np.flipud(self._canonical_positions),
            5: np.rot90(np.flipud(self._canonical_positions), axes=(0, 1)),
            6: np.rot90(np.flipud(self._canonical_positions), k=2, axes=(0, 1)),
            7: np.rot90(np.flipud(self._canonical_positions), k=3, axes=(0, 1)),
        }

    def get_transformed_states(self, game) -> list['InvestigateGame']:
        '''
        Apply all possible transformations to the state and return a list of equivalent transformed states.
        To compute all equivalent states, apply:
        - a 90° rotation to the original state;
        - a 180° rotation to the original state;
        - a 270° rotation to the original state;
        - a up/down flip;
        - a 90° rotation to the flipped state;
        - a 180° rotation to the flipped state;
        - a 270° rotation to the flipped state.

        Args:
            None.

        Returns:
            A list of equivalent transformed states is returned.
        '''

        # define list of transformed states
        transformed_states = []

        # for each flip
        for flip in self._flips:
            # copy of the state
            new_state = deepcopy(game)
            # transform the board
            flipped_board = flip(game._board)
            # new flipped state
            new_state._board = flipped_board
            # append transformed state
            transformed_states.append(new_state)
            # add rotated states
            for rotate in self._rotations:
                # copy of the state
                rotated_new_state = deepcopy(new_state)
                # transform the board
                rotated_new_state._board = rotate(new_state._board)
                # append transformed state
                transformed_states.append(rotated_new_state)

        return transformed_states

    def get_action_from_canonical_action(self, action: tuple[tuple[int, int], Move], transformation_index: int) -> tuple[tuple[int, int], Move]:
        '''
        Gives the corresponding action for a state compared to the canonical state.

        Args:
            action: the canonical action,
            transformation_index: the corrisponding index of the trasformation applied.

        Returns:
            The corrisponding action is returned.
        '''
        if transformation_index == 0:
            return action

        return (
            self._trasformed_positions[transformation_index][action[0]],
            self._map_canonical_slide_to_original_slide[transformation_index][action[1]],
        )
