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

    rotations = [lambda x: np.rot90(x, k=1), lambda x: np.rot90(x, k=2), lambda x: np.rot90(x, k=3)]
    flips = [lambda x: deepcopy(x), lambda x: np.flipud(x)]
    # mapping of canonical slide to original slide
    map_canonical_slide_to_original_slide = {
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
    canonical_positions = np.array(
        [
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
            [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
            [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)],
            [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
        ]
    )
    # mapping trasformation - trasformed from position
    trasformed_positions = {
        1: np.rot90(canonical_positions, k=1),
        2: np.rot90(canonical_positions, k=2),
        3: np.rot90(canonical_positions, k=3),
        4: np.flipud(canonical_positions),
        5: np.rot90(np.flipud(canonical_positions), k=1),
        6: np.rot90(np.flipud(canonical_positions), k=2),
        7: np.rot90(np.flipud(canonical_positions), k=3),
    }

    @classmethod
    def get_transformed_states(cls, game) -> list['InvestigateGame']:
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
        for flip in Symmetry.flips:
            # copy of the state
            new_state = deepcopy(game)
            # transform the board
            flipped_board = flip(new_state._board)
            # new flipped state
            new_state._board = flipped_board
            # append transformed state
            transformed_states.append(new_state)
            # add rotated states
            for rotate in Symmetry.rotations:
                # copy of the state
                rotated_new_state = deepcopy(new_state)
                # transform the board
                rotated_new_state._board = rotate(rotated_new_state._board)
                # append transformed state
                transformed_states.append(rotated_new_state)

        return transformed_states

    @classmethod
    def get_action_from_canonical_action(cls, action: tuple[tuple[int, int], Move], transformation_index: int) -> tuple[tuple[int, int], Move]:
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

        from_pos, slide = action

        return (
            tuple(Symmetry.trasformed_positions[transformation_index][(from_pos[1], from_pos[0])]),
            Symmetry.map_canonical_slide_to_original_slide[transformation_index][slide],
        )
