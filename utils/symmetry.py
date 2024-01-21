from game import Move
import numpy as np
from copy import deepcopy


class Symmetry:
    '''
    Class that handles information about the symmetries of the board.

    The mapping from index to transformation is the following:
        0 : identity
        1 : rot90(  )
        2 : rot180(  )
        3 : rot270(  )
        4 : flipud(  )
        5 : rot90( flipud(  ) )
        6 : rot180( flipud(  ) )
        7 : rot270( flipud(  ) )
    '''

    # define the rotations
    rotations = [lambda x: np.rot90(x, k=1), lambda x: np.rot90(x, k=2), lambda x: np.rot90(x, k=3)]
    # define the flips
    flips = [lambda x: deepcopy(x), lambda x: np.flipud(x)]
    # define the swaps
    swaps = [lambda x: deepcopy(x), lambda x: Symmetry.__swap_board_players(x)]

    # define a matrix to indicate relevant directions
    compass = np.array([['', Move.TOP, ''], [Move.LEFT, '', Move.RIGHT], ['', Move.BOTTOM, '']])

    # map a move to indices for the compass
    map_slide_to_compass = {
        Move.TOP: (0, 1),
        Move.LEFT: (1, 0),
        Move.RIGHT: (1, 2),
        Move.BOTTOM: (2, 1),
    }

    # map a slide to a canonical slide
    trasformation_to_canonical_slides = {
        1: np.rot90(compass, k=1),
        2: np.rot90(compass, k=2),
        3: np.rot90(compass, k=3),
        4: np.flipud(compass),
        5: np.rot90(np.flipud(compass), k=1),
        6: np.rot90(np.flipud(compass), k=2),
        7: np.rot90(np.flipud(compass), k=3),
    }

    # map a canonical slide to a slide
    trasformation_to_non_canonnical_slides = {
        1: np.rot90(compass, k=-1),
        2: np.rot90(compass, k=-2),
        3: np.rot90(compass, k=-3),
        4: np.flipud(compass),
        5: np.flipud(np.rot90(compass, k=-1)),
        6: np.flipud(np.rot90(compass, k=-2)),
        7: np.flipud(np.rot90(compass, k=-3)),
    }

    # define the board positions
    positions = np.array(
        [
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
            [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
            [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)],
            [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
        ]
    )

    # map a position to a canonical position
    trasformation_to_canonical_positions = {
        1: np.rot90(positions, k=1),
        2: np.rot90(positions, k=2),
        3: np.rot90(positions, k=3),
        4: np.flipud(positions),
        5: np.rot90(np.flipud(positions), k=1),
        6: np.rot90(np.flipud(positions), k=2),
        7: np.rot90(np.flipud(positions), k=3),
    }

    # map a canonical position to a position
    trasformation_to_non_canonnical_positions = {
        1: np.rot90(positions, k=-1),
        2: np.rot90(positions, k=-2),
        3: np.rot90(positions, k=-3),
        4: np.flipud(positions),
        5: np.flipud(np.rot90(positions, k=-1)),
        6: np.flipud(np.rot90(positions, k=-2)),
        7: np.flipud(np.rot90(positions, k=-3)),
    }

    @classmethod
    def get_canonical_state(cls, game: 'InvestigateGame', player_id: int = None) -> int:
        '''
        Apply all possible transformations to the state and return the canonical state representation.
        To compute all equivalent states, apply:
            - a 90° rotation to the original state;
            - a 180° rotation to the original state;
            - a 270° rotation to the original state;
            - a up/down flip;
            - a 90° rotation to the flipped state;
            - a 180° rotation to the flipped state;
            - a 270° rotation to the flipped state.
        These trasformation are applied to the original state and the swapped state.

        Args:
            game: a game instance;
            player_id: the player's id.

        Returns:
            The canonical state representation is returned.
        '''

        # define list of transformed states
        transformed_states = []

        # for each swap
        for idx, swap in enumerate(Symmetry.swaps):
            # copy the state
            swapped_state = deepcopy(game)
            # swap the board
            swapped_state._board = swap(swapped_state._board)
            # for each flip
            for flip in Symmetry.flips:
                # copy the state
                flipped_state = deepcopy(swapped_state)
                # transform the board
                flipped_state._board = flip(flipped_state._board)
                # append the transformed state
                transformed_states.append(flipped_state.get_hashable_state(player_id if idx == 0 else 1 - player_id))
                # for each rotation
                for rotate in Symmetry.rotations:
                    # copy the state
                    rotated_new_state = deepcopy(flipped_state)
                    # transform the board
                    rotated_new_state._board = rotate(rotated_new_state._board)
                    # append the transformed state
                    transformed_states.append(
                        rotated_new_state.get_hashable_state(player_id if idx == 0 else 1 - player_id)
                    )

        return min(transformed_states)

    @classmethod
    def get_action_from_canonical_action(
        cls, action: tuple[tuple[int, int], Move], transformation_index: int
    ) -> tuple[tuple[int, int], Move]:
        '''
        Gives the corresponding action for a state compared to the canonical state.

        Args:
            action: the canonical action;
            transformation_index: the corresponding index of the applied trasformation.

        Returns:
            The corresponding action is returned.
        '''
        # if no transformation is applied
        if transformation_index == 0:
            # return the action as it is
            return action

        # unpack the action
        from_pos, slide = action

        # return the transformed action
        return (
            tuple(Symmetry.trasformation_to_canonical_positions[transformation_index][(from_pos[1], from_pos[0])]),
            Symmetry.trasformation_to_canonical_slides[transformation_index][Symmetry.map_slide_to_compass[slide]],
        )

    @classmethod
    def get_canonical_action_from_action(
        cls, action: tuple[tuple[int, int], Move], transformation_index: int
    ) -> tuple[tuple[int, int], Move]:
        '''
        Gives the corresponding canonical action for a state which is not canonical.

        Args:
            action: the action;
            transformation_index: the corrisponding index of the trasformation applied.

        Returns:
            The corrisponding canonical action is returned.
        '''
        # if no transformation is applied
        if transformation_index == 0:
            return action

        # unpack the action
        from_pos, slide = action

        # return the transformed action
        return (
            tuple(Symmetry.trasformation_to_non_canonnical_positions[transformation_index][(from_pos[1], from_pos[0])]),
            Symmetry.trasformation_to_non_canonnical_slides[transformation_index][Symmetry.map_slide_to_compass[slide]],
        )

    @classmethod
    def __swap_board_players(cls, board: np.ndarray) -> np.ndarray:
        """
        Trasform the board of the game by swapping the players.

        Args:
            board: the board game.

        Returns:
            A board with the players swapped is returned.
        """
        # trasform the board by swapping the players
        tmp = board == 0
        board[board == 1] = 0
        board[tmp] = 1

        return board
