import numpy as np
import time
from game import Game, Move, Player
from investigate_game import InvestigateGame
from collections import defaultdict
import pickle
from collections import namedtuple

EntryMinMax = namedtuple('EntryMinMax', ['depth', 'value'])
EntryAlphaBetaMinMax = namedtuple('EntryMinMax', ['depth', 'value', 'action'])


class MinMaxPlayer(Player):
    """
    Class representing a player which plays according to the Min-Max algorithm.
    """

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
        self._visited_max_states = {}
        self._visited_min_states = {}
        self._hit = 0

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
        # get hashable state
        key = game.get_hashable_state(self._player_id)

        # check if this max_value is already in hash table
        if key in self._visited_max_states and depth <= self._visited_max_states[key].depth:
            self._hit += 1
            return self._visited_max_states[key].value

        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get terminal value
            value = self.evaluation_function(game)
            # save max_value in hash_table
            self._visited_max_states[key] = EntryMinMax(0, value)
            # return its heuristic value
            return self.evaluation_function(game)
        # set the current best max value
        value = float('-inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions(self._player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._player_id)
        )
        # for each possible game transitions
        for _, state in transitions:
            # update the current max value
            value = max(value, self.min_value(state, depth - 1))

        # save max_value in hash_table
        self._visited_max_states[key] = EntryMinMax(depth, value)
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
        # get hashable state
        key = game.get_hashable_state(self._player_id)

        # check if this max_value is already in hash table
        if key in self._visited_min_states and depth <= self._visited_min_states[key].depth:
            self._hit += 1
            return self._visited_min_states[key].value

        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get terminal value
            value = self.evaluation_function(game)
            # save min_value in hash_table
            self._visited_min_states[key] = EntryMinMax(0, value)
            # return its heuristic value
            return value
        # set the current best min value
        value = float('inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions(self._opponent_player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._opponent_player_id)
        )
        # for each possible game transitions
        for _, state in transitions:
            # update the current min value
            value = min(value, self.max_value(state, depth - 1))

        # save min_value in hash_table
        self._visited_min_states[key] = EntryMinMax(depth, value)
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
        game = InvestigateGame(game)
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions(self._player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._player_id)
        )
        # for all possible actions and result states
        actions, states = zip(*transitions)
        # return the action corresponding to the best estimated move
        tuple_ = max(enumerate(actions), key=lambda t: self.min_value(states[t[0]], self._depth - 1))
        # extract the action
        _, action = tuple_
        # return it
        return action

    def save(self, path: str) -> None:
        """
        Serialize the current MinMax player's state.

        Args:
            path: location where to save the player's state.

        Returns: None.
        """
        # serialize the Monte Carlo learning player
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """
        Load a MinMax player's state into the current player.

        Args:
            path: location from which to load the player's state.

        Returns: None.
        """
        # load the serialized Monte Carlo learning player
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)


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

    def __default_value(self) -> tuple[0, None]:
        """
        Implement a default factoty method for defaultdic.

        Args:
            None.

        Returns:
            A default value is returned.
        """
        return (0, None)

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

        # get hashable state
        key = game.get_hashable_state(self._player_id)

        # check if this max_value is already in hash table
        if key in self._visited_max_states and depth <= self._visited_max_states[key].value:
            self._hit += 1
            return self._visited_max_states[key].value

        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get terminal value
            value = self.evaluation_function(game)
            # save min_value in hash_table
            self._visited_max_states[key] = EntryAlphaBetaMinMax(0, value, None)
            # return its heuristic value and no move
            return value, None

        # set the current best max value
        best_value = float('-inf')
        # set the current best move
        best_action = None
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions(self._player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._player_id)
        )
        # for each possible game transitions
        for action, state in transitions:
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
                # save min_value in hash_table
                self._visited_max_states[key] = EntryAlphaBetaMinMax(depth, best_value, best_action)
                # terminate the search
                return best_value, best_action

        # save max_value in hash_table
        self._visited_max_states[key] = EntryAlphaBetaMinMax(depth, best_value, best_action)
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
        # get hashable state
        key = game.get_hashable_state(self._player_id)

        # check if this max_value is already in hash table
        if key in self._visited_min_states and depth <= self._visited_min_states[key].value:
            self._hit += 1
            return self._visited_min_states[key].value

        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get terminal value
            value = self.evaluation_function(game)
            # save min_value in hash_table
            self._visited_min_states[key] = EntryAlphaBetaMinMax(0, value, None)
            # return its heuristic value and no move
            return value, None

        # set the current best min value
        best_value = float('inf')
        # set the current best move
        best_action = None
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions(self._opponent_player_id)
            if self._symmetries
            else game.generate_possible_transitions(self._opponent_player_id)
        )
        # for each possible game transitions
        for action, state in transitions:
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
                # save min_value in hash_table
                self._visited_min_states[key] = EntryAlphaBetaMinMax(depth, best_value, best_action)
                # terminate the search
                return best_value, best_action

        # save min_value in hash_table
        self._visited_min_states[key] = EntryAlphaBetaMinMax(depth, best_value, best_action)
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
        game = InvestigateGame(game)
        # get the best move to play
        _, action = self.max_value(game, self._depth, float('-inf'), float('inf'))
        # return it
        return action


if __name__ == '__main__':
    # test generate_canonical_transitions() execution time vs generate_possible_transitions()
    game = Game()
    start = time.time()
    transictions = AlphaBetaMinMaxPlayer(1).InvestigateGame(game).generate_canonical_transitions(1)
    print(len(transictions))
    total_time = time.time() - start
    print(f'Canonical: {total_time:.2E} sec, {total_time / 60:.2E} min')

    start = time.time()
    transictions = AlphaBetaMinMaxPlayer(1).InvestigateGame(game).generate_possible_transitions(1)
    print(len(transictions))
    total_time = time.time() - start
    print(f'Non canonical: {total_time:.2E} sec, {total_time / 60:.2E} min')
