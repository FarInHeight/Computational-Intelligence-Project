import numpy as np
import time
from game import Game, Move, Player
from investigate_game import InvestigateGame
import pickle
from collections import namedtuple
from joblib import Parallel, delayed

EntryMinMax = namedtuple('EntryMinMax', ['depth', 'value'])


class MinMaxPlayer(Player):
    """
    Class representing a player which plays according to the Min-Max algorithm.
    """

    def __init__(self, depth: int = 3, symmetries: bool = False, enhance: bool = False) -> None:
        """
        Constructor of the Min-Max player.

        Args:
            depth: maximum depth of the Min-Max search tree;
            symmetries: flag to consider the symmetries or not;
            enhance: choose whether to enhance the evaluation function.

        Returns:
            None.
        """
        super().__init__()
        self._depth = depth
        self._symmetries = symmetries
        self._enhance = enhance
        self._visited_max_states = {}
        self._visited_min_states = {}
        self._hit = 0

    def max_value(self, game: 'InvestigateGame', key: int, depth: int) -> int | float:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Max player to a maximum depth.

        Args:
            game: the current game state;
            key: the current game state representation;
            depth: the current depth in the search tree.

        Returns:
            The evaluation function value of the best move to play
            for Max is returned.
        """
        # check if this max_value is already in hash table
        if key in self._visited_max_states and depth <= self._visited_max_states[key].depth:
            self._hit += 1
            return self._visited_max_states[key].value

        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get terminal value
            value = game.evaluation_function(game.current_player_idx, self._enhance)
            # save max_value in hash_table
            self._visited_max_states[key] = EntryMinMax(0, value)
            # return its heuristic value
            return value
        # set the current best max value
        value = float('-inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # for each possible game transitions
        for _, state, key in transitions:
            # update the current max value
            value = max(value, self.min_value(state, key, depth - 1))

        # save max_value in hash_table
        self._visited_max_states[key] = EntryMinMax(depth, value)
        return value

    def min_value(self, game: 'InvestigateGame', key: int, depth: int) -> int | float:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Min player to a maximum depth.

        Args:
            game: the current game state;
            key: the current game state representation;
            depth: the current depth in the search tree.

        Returns:
            The evaluation function value of the best move to play
            for Min is returned.
        """
        # check if this max_value is already in hash table
        if key in self._visited_min_states and depth <= self._visited_min_states[key].depth:
            self._hit += 1
            return self._visited_min_states[key].value

        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get terminal value
            value = game.evaluation_function(1 - game.current_player_idx, self._enhance)
            # save min_value in hash_table
            self._visited_min_states[key] = EntryMinMax(0, value)
            # return its heuristic value
            return value
        # set the current best min value
        value = float('inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # for each possible game transitions
        for _, state, key in transitions:
            # update the current min value
            value = min(value, self.max_value(state, key, depth - 1))

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
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # parallelize min_value
        values = Parallel(n_jobs=-1)(
            delayed(self.min_value)(state, key, self._depth - 1) for _, state, key in transitions
        )
        # return the action corresponding to the best estimated move
        _, (action, _, _) = max(
            enumerate(transitions),
            key=lambda t: values[t[0]],
        )
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

    def __init__(self, depth: int = 3, symmetries: bool = False, enhance: bool = False) -> None:
        """
        Constructor of the Min-Max + Alpha-Beta pruning player.

        Args:
            depth: maximum depth of the Min-Max search tree;
            symmetries: flag to consider the symmetries or not;
            enhance: choose whether to enhance the evaluation function.

        Returns:
            None.
        """
        super().__init__(depth, symmetries, enhance)

    def max_value(
        self, game: 'Game', key: int, depth: int, alpha: float, beta: float
    ) -> tuple[int | float, None | tuple[tuple[int, int], Move]]:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Max player to a maximum depth by cutting off
        some branches whenever a Min ancestor cannot improve
        its associated value.

        Args:
            game: the current game state;
            key: the current game state representation;
            depth: the current depth in the search tree;
            alpha: the best value among all Max ancestors;
            beta: the best value among all Min ancestors.

        Returns:
            The evaluation function value of the best move to play
            for Max and the move itsef are returned.
        """
        # check if this max_value is already in hash table
        if key in self._visited_max_states and depth <= self._visited_max_states[key].depth:
            self._hit += 1
            return self._visited_max_states[key].value

        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get terminal value
            value = game.evaluation_function(game.current_player_idx, self._enhance)
            # save min_value in hash_table
            self._visited_max_states[key] = EntryMinMax(0, value)
            # return its heuristic value and no move
            return value

        # set the current best max value
        best_value = float('-inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # for each possible game transitions
        for _, state, key in transitions:
            print(state.current_player_idx)
            # play as Min
            value = self.min_value(state, key, depth - 1, alpha, beta)
            # if we find a better value
            if value > best_value:
                # update the current max value
                best_value = value
                # update the maximum Max value so far
                alpha = max(alpha, best_value)
            # if the value for the best Min ancestor cannot be improved
            if best_value >= beta:
                # save min_value in hash_table
                self._visited_max_states[key] = EntryMinMax(depth, best_value)
                # terminate the search
                return best_value

        # save max_value in hash_table
        self._visited_max_states[key] = EntryMinMax(depth, best_value)
        return best_value

    def min_value(
        self, game: 'Game', key: int, depth: int, alpha: float, beta: float
    ) -> tuple[int | float, None | tuple[tuple[int, int], Move]]:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Min player to a maximum depth by cutting off
        some branches whenever a Max ancestor cannot improve
        its associated value.

        Args:
            game: the current game state;
            key: the current game state representation;
            depth: the current depth in the search tree;
            alpha: the best value among all Max ancestors;
            beta: the best value among all Min ancestors.

        Returns:
            The evaluation function value of the best move to play
            for Min and the move itsef are returned.
        """
        # check if this max_value is already in hash table
        if key in self._visited_min_states and depth <= self._visited_min_states[key].depth:
            self._hit += 1
            return self._visited_min_states[key].value

        # if there are no more levels to examinate or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get terminal value
            value = game.evaluation_function(1 - game.current_player_idx, self._enhance)
            # save min_value in hash_table
            self._visited_min_states[key] = EntryMinMax(0, value)
            # return its heuristic value and no move
            return value

        # set the current best min value
        best_value = float('inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # for each possible game transitions
        for _, state, key in transitions:
            # play as Max
            value = self.max_value(state, key, depth - 1, alpha, beta)
            # if we find a better value
            if value < best_value:
                # update the current min value
                best_value = value
                # update the minimum Min value so far
                beta = min(beta, best_value)
            # if the value for the best Max ancestor cannot be improved
            if best_value <= alpha:
                # save min_value in hash_table
                self._visited_min_states[key] = EntryMinMax(depth, best_value)
                # terminate the search
                return best_value

        # save min_value in hash_table
        self._visited_min_states[key] = EntryMinMax(depth, best_value)
        return best_value

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
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # parallelize min_value
        values = Parallel(n_jobs=-1)(
            delayed(self.min_value)(state, key, self._depth - 1, float('-inf'), float('inf'))
            for _, state, key in transitions
        )
        # return the action corresponding to the best estimated move
        _, (action, _, _) = max(
            enumerate(transitions),
            key=lambda t: values[t[0]],
        )
        # return it
        return action


if __name__ == '__main__':
    from tqdm import trange
    from random_player import RandomPlayer

    def test(player1, player2, num_games, idx):
        wins = 0
        pbar = trange(num_games)
        for game in pbar:
            g = Game()
            w = g.play(player1, player2)
            if w == idx:
                wins += 1
            pbar.set_description(f'Current percentage of wins player {idx}: {wins/(game+1):%}')
        print(f'Percentage of wins player {idx}: {wins/num_games:%}')

    print(f'AlphaBetaMinMax as first')
    test(
        MinMaxPlayer(depth=2, symmetries=False, enhance=False),
        RandomPlayer(),
        10,
        0,
    )
    print(f'AlphaBetaMinMax as second')
    test(
        RandomPlayer(),
        MinMaxPlayer(depth=2, symmetries=False, enhance=False),
        10,
        1,
    )
