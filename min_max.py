import numpy as np
from game import Game, Move, Player
from copy import deepcopy
from itertools import product


class MinMaxPlayer(Player):
    """
    Class representing a player which plays according to the Min-Max algorithm.
    """

    class InvestigateGame(Game):
        """
        Class representing an extension of the Game class which
        is used by the Min-Man players to construct the search tree.
        """

        def __init__(self, game: 'Game', symmetries=False) -> None:
            """
            Constructor for creating a copy of the game to investigate.

            Args:
                game: the current game state;
                symmetries: flag to consider the symmetries or not.

            Returns:
                None.
            """
            super().__init__()
            self._board = game.get_board()
            self._rotations = [lambda x: np.rot90(x, k=1), lambda x: np.rot90(x, k=2), lambda x: np.rot90(x, k=3)]
            self._flips = [lambda x: np.fliplr(x), lambda x: np.flipud(x)]
            self._symmetries = symmetries

        def __eq__(self, other) -> bool:
            '''Equality check for the class. Are equal if the boards are equal.'''
            return (self._board == other._board).all()

        def _get_transformed_states(self) -> list['InvestigateGame']:
            '''
            Apply all possible transformations to the state and return a list of possible transformed states.
            To compute all equivalent states, apply a 90°, 180°, and 270° rotation.
            A horizontal and vertical flip, and a 90° rotation on the two flipped states.
            Because a 180° rotation of the flipped states is equivalent to a flip in the original position,
            and a 270° rotation of the flipped states is equivalent to flipping the 90° rotation.
            (270° rotation of horizontal/vertical flip = 90° rotation of vertical/horizontal flip) and
            (horizontal/vertical flip = 180° rotation of vertical/horizontal flip)
            '''

            # init list of transformed states
            transformed_states = []

            # adding flipped and flipped + rotate90
            for flip in self._flips:
                # copy of the state
                state = deepcopy(self)
                # transform the board
                flipped_board = flip(state._board)
                # new flipped state
                state._board = flipped_board
                # append transformed state
                transformed_states.append(state)
                # new state flipped and rotate
                state._board = np.rot90(flipped_board)
                # append new transformed state
                transformed_states.append(state)

            # adding rotated
            for rotate in self._rotations:
                # copy of the state
                state = deepcopy(self)
                # transform the board
                state._board = rotate(state._board)
                # append transformed state
                transformed_states.append(state)

            # if canonic state is in transformed states
            if self in transformed_states:
                # remove the canonic state
                transformed_states.remove(self)

            return transformed_states

        def generate_possible_transitions(self, player_id: int) -> list[tuple[tuple[int, int], Move]]:
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

        def generate_canonical_transitions(self, player_id: int) -> list[tuple[tuple[int, int], Move]]:
            """
            Generate all possible game transitions that a given player can make considering the canonic states.

            Args:
                player_id: the player's id.

            Returns:
                A list of pairs of actions and corresponding game states
                is returned.
            """

            # get possible transictions
            transictions = self.generate_possible_transitions(player_id)
            # get the states
            _, states = zip(*transictions)
            # convert to list
            states = list(states)

            i = 0
            # for each transiction
            while i < len(states):
                # compute all the transformed states
                transformed_states = states[i]._get_transformed_states()
                # delete transformed states
                states = [state for state in states if state not in transformed_states]
                # increment index
                i += 1

            return [transiction for transiction in transictions if transiction[1] in states]

    def __init__(self, player_id: int, depth: int = 3, symmetries: bool = False) -> None:
        """
        Constructor of the Min-Max player.

        Args:
            player_id: the player's id;
            depth: maximum depth of the Min-Max search tree;
            symmetries: flag to consider the symmetries or not.

        Returns:
            None.
        """
        super().__init__()
        self._player_id: int = player_id
        self._opponent_player_id = abs(1 - self._player_id)
        self._depth = depth
        self._symmetries = symmetries

    def evaluation_function(self, game: 'InvestigateGame') -> int | float:
        """
        Given the current state of the game, a static evaluation is performed
        and it is determined if the current position is an advantageous one
        for the Max player or the Min player.
        Values greater than zero indicate that Max will probably win,
        zero indicates a balanced situation and values lower than
        zero indicate that Min will probably win.
        The value is calculated as:
        (number of complete rows, columns, or diagonals that are stil open
        for MAX) - (number of complete rows, columns, or diagonals that are
        stil open for MIN)

        Args:
            game: the current game state.

        Return:
            An estimate value of how much a current player is winning
            or losing is returned.
        """

        # check if the game is over
        value = game.check_winner()
        # take the max player id
        max_player = self._player_id
        # take the min player id
        min_player = self._opponent_player_id

        # if max wins return +inf
        if value == max_player:
            return float('inf')
        # if min wins return -inf
        elif value == min_player:
            return float('-inf')

        # turn each player id into a type compatible with the game board
        max_player = np.int16(max_player)
        min_player = np.int16(min_player)
        neutral_pos = np.int16(-1)

        # define the max value
        max_value = 0
        # define the min value
        min_value = 0
        # get the board
        board = game.get_board()

        # define which board lines to examinate
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
            # take the max piece positions
            max_taken_positions = line == max_player
            # take the min piece positions
            min_taken_positions = line == min_player
            # take the neutral piece positions
            neutral_positions = line == neutral_pos
            # if all the pieces are neutral or belong to the max player
            if all(np.logical_or(max_taken_positions, neutral_positions)):
                # increment the max value
                max_value += 1
            # if all the pieces are neutral or belong to the min player
            if all(np.logical_or(min_taken_positions, neutral_positions)):
                # increment the min value
                min_value += 1

        return max_value - min_value

    def max_value(self, game: 'InvestigateGame', depth: int) -> int | float:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Max player to a maximum depth.

        Args:
            game: the current game state;
            depth: the current depth in the search tree.

        Returns:
            The evaluation function value of the best move to play
            for Max is returned.
        """
        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # return its heuristic value
            return self.evaluation_function(game)
        # set the current best max value
        value = float('-inf')
        # get all possible game transictions or canonical transictions
        transictions = (
            game.generate_canonical_transitions(self._player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._player_id)
        )
        # for each possible game transictions
        for _, state in transictions:
            # update the current max value
            value = max(value, self.min_value(state, depth - 1))
        return value

    def min_value(self, game: 'InvestigateGame', depth: int) -> int | float:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Min player to a maximum depth.

        Args:
            game: the current game state;
            depth: the current depth in the search tree.

        Returns:
            The evaluation function value of the best move to play
            for Min is returned.
        """

        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # return its heuristic value
            return self.evaluation_function(game)
        # set the current best min value
        value = float('inf')
        # get all possible game transictions or canonical transictions
        transictions = (
            game.generate_canonical_transitions(self._opponent_player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._opponent_player_id)
        )
        # for each possible game transictions
        for _, state in transictions:
            # update the current min value
            value = min(value, self.max_value(state, depth - 1))
        return value

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Implement the MinMax procedure.

        Args:
            game: the current game state.

        Returns:
            The best move to play for Max is returned.
        """
        # create seperate instance of a game for investigation
        game = self.InvestigateGame(game, self._symmetries)
        # get all possible game transictions or canonical transictions
        transictions = (
            game.generate_canonical_transitions(self._opponent_player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._opponent_player_id)
        )
        # for all possible actions and result states
        actions, states = zip(*transictions)
        # return the action corresponding to the best estimated move
        tuple_ = max(enumerate(actions), key=lambda t: self.min_value(states[t[0]], self._depth - 1))
        # extract the action
        _, action = tuple_
        # return it
        return action


class AlphaBetaMinMaxPlayer(MinMaxPlayer):
    """
    Class representing a player which plays according to the
    Min-Max algorithm improved by the alpha-beta pruning technique.
    """

    def __init__(self, player_id: int, depth: int = 3, symmetries: bool = False) -> None:
        """
        Constructor of the Min-Max + Alpha-Beta pruning player.

        Args:
            player_id: the player's id;
            depth: maximum depth of the Min-Max search tree;
            symmetries: flag to consider the symmetries or not.

        Returns:
            None.
        """
        super().__init__(player_id, depth, symmetries)

    def max_value(
        self, game: 'Game', depth: int, alpha: float, beta: float
    ) -> tuple[int | float, None | tuple[tuple[int, int], Move]]:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Max player to a maximum depth by cutting off
        some branches whenever a Min ancestor cannot improve
        its associated value.

        Args:
            game: the current game state;
            depth: the current depth in the search tree;
            alpha: the best value among all Max ancestors;
            beta: the best value among all Min ancestors.

        Returns:
            The evaluation function value of the best move to play
            for Max and the move itsef are returned.
        """
        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # return its heuristic value and no move
            return self.evaluation_function(game), None
        # set the current best max value
        best_value = float('-inf')
        # set the current best move
        best_action = None
        # get all possible game transictions or canonical transictions
        transictions = (
            game.generate_canonical_transitions(self._player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._player_id)
        )
        # for each possible game transictions
        for action, state in transictions:
            # play as Min
            value, _ = self.min_value(state, depth - 1, alpha, beta)
            # if we find a better value
            if value > best_value:
                # update the current max value
                best_value = value
                # update the current best move
                best_action = action
                # update the maximum Max value so far
                alpha = max(alpha, best_value)
            # if the value for the best Min ancestor cannot be improved
            if best_value >= beta:
                # terminate the search
                return best_value, best_action
        return best_value, best_action

    def min_value(
        self, game: 'Game', depth: int, alpha: float, beta: float
    ) -> tuple[int | float, None | tuple[tuple[int, int], Move]]:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Min player to a maximum depth by cutting off
        some branches whenever a Max ancestor cannot improve
        its associated value.

        Args:
            game: the current game state;
            depth: the current depth in the search tree;
            alpha: the best value among all Max ancestors;
            beta: the best value among all Min ancestors.

        Returns:
            The evaluation function value of the best move to play
            for Min and the move itsef are returned.
        """
        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # return its heuristic value and no move
            return self.evaluation_function(game), None
        # set the current best min value
        best_value = float('inf')
        # set the current best move
        best_action = None
        # get all possible game transictions or canonical transictions
        transictions = (
            game.generate_canonical_transitions(self._opponent_player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._opponent_player_id)
        )
        # for each possible game transictions
        for action, state in transictions:
            # play as Max
            value, _ = self.max_value(state, depth - 1, alpha, beta)
            # if we find a better value
            if value < best_value:
                # update the current min value
                best_value = value
                # update the current best move
                best_action = action
                # update the minimum Min value so far
                beta = min(beta, best_value)
            # if the value for the best Max ancestor cannot be improved
            if best_value <= alpha:
                # terminate the search
                return best_value, best_action
        return best_value, best_action

    def make_move(self, game: 'Game') -> tuple[int | float, None | tuple[tuple[int, int], Move]]:
        """
        Implement the MinMax procedure with alpha-beta pruning.

        Args:
            game: the current game state.

        Returns:
            The best move to play for Max is returned.
        """
        # create seperate instance of a game for investigation
        game = self.InvestigateGame(game, self._symmetries)
        # get the best move to play
        _, action = self.max_value(game, self._depth, float('-inf'), float('inf'))
        # return it
        return action
