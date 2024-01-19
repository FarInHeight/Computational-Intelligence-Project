from game import Game, Move, Player
from utils.investigate_game import InvestigateGame
import pickle
from collections import namedtuple
from joblib import Parallel, delayed
from tqdm import trange
from players.random_player import RandomPlayer
from copy import deepcopy


EntryMinMax = namedtuple('EntryMinMax', ['depth', 'value'])


class MinMaxPlayer(Player):
    """
    Class representing a player which plays according to the Min-Max algorithm.
    """

    def __init__(self, depth: int = 3, symmetries: bool = False, enhance: bool = True) -> None:
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
        self._visited = {}
        self._depth = depth
        self._symmetries = symmetries
        self._enhance = enhance
        self._parallelize = True

    def parallelize(self, activate: bool = True) -> None:
        """
        Decide if to parallelize the MinMax procedure. If parallelization is activated,
        then the hash table is deleted.

        Args:
            activate: whether to activate the parallelization or not.

        Returns:
            None.
        """
        # set parallelization flag
        self._parallelize = activate
        if activate:
            # reset visited dict state
            self._visited = {}

    def max_value(self, game: 'InvestigateGame', key: int, depth: int) -> int | float:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Max player to a maximum depth.

        Args:
            game: the current game state;
            key: the current game state representation;
            depth: the remaining depth in the search tree.

        Returns:
            The evaluation function value of the best move to play
            for Max is returned.
        """
        # check if the state is already in hash table
        if key in self._visited and depth <= self._visited[key].depth:
            return self._visited[key].value

        # if there are no more levels to examine or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get the heuristic value of the state
            value = game.evaluation_function(game.current_player_idx, self._enhance)
            # save the state in hash table
            self._visited[key] = EntryMinMax(0, value)
            # return its heuristic value
            return value
        # set the current best max value
        value = float('-inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # for each possible game transition
        for _, state, key in transitions:
            # update the current max value
            value = max(value, self.min_value(state, key, depth - 1))

        # save the state in hash table
        self._visited[key] = EntryMinMax(depth, value)
        return value

    def min_value(self, game: 'InvestigateGame', key: int, depth: int) -> int | float:
        """
        Perform a recursive traversal of the adversarial search tree
        for the Min player to a maximum depth.

        Args:
            game: the current game state;
            key: the current game state representation;
            depth: the remaining depth in the search tree.

        Returns:
            The evaluation function value of the best move to play
            for Min is returned.
        """
        # check if the state is already in hash table
        if key in self._visited and depth <= self._visited[key].depth:
            return self._visited[key].value

        # if there are no more levels to examine or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get the heuristic value
            value = game.evaluation_function(1 - game.current_player_idx, self._enhance)
            # save the state in hash table
            self._visited[key] = EntryMinMax(0, value)
            # return its heuristic value
            return value
        # set the current best min value
        value = float('inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # for each possible game transition
        for _, state, key in transitions:
            # update the current min value
            value = min(value, self.max_value(state, key, depth - 1))

        # save the state in hash table
        self._visited[key] = EntryMinMax(depth, value)
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
        # get first canonical level
        transitions = game.generate_canonical_transitions()
        # if we are parallelizing
        if self._parallelize:
            # parallelize min_value
            values = Parallel(n_jobs=-1)(
                delayed(self.min_value)(state, key, self._depth - 1) for _, state, key in transitions
            )
        # otherwise
        else:
            # do not parallelize
            values = [self.min_value(state, key, self._depth - 1) for _, state, key in transitions]

        # return the action corresponding to the best estimated move
        _, (action, _, _) = max(
            enumerate(transitions),
            key=lambda t: values[t[0]],
        )
        # return it
        return action

    def train(self, n_games: int = 100) -> None:
        """
        Train the MinMax player by updating the hash table.

        Args:
            n_games: number of games to play against a random player;

        Returns:
            None.
        """
        # set parallelize flag to false
        self._parallelize = False
        # get current number of found states
        initial_n_states = len(self._visited)
        # define the players
        players = self, RandomPlayer()
        # define how many game to play
        pbar = trange(n_games)
        # for each game
        for _ in pbar:
            # instantiate the game
            g = Game()
            # play the game
            g.play(*players)
            # swap the players
            players = players[1], players[0]
            pbar.set_description(f"Found states: {len(self._visited):,}")
        print(f"New found states after {n_games} games: {(len(self._visited) - initial_n_states):,}")

    def save(self, path: str) -> None:
        """
        Serialize the current MinMax player's state.

        Args:
            path: location where to save the player's state.

        Returns: None.
        """
        # serialize the MinMax player
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """
        Load a MinMax player's state into the current player.

        Args:
            path: location from which to load the player's state.

        Returns: None.
        """
        # load the serialized MinMax player
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
            depth: the remaining depth in the search tree;
            alpha: the best value among all Max ancestors;
            beta: the best value among all Min ancestors.

        Returns:
            The evaluation function value of the best move to play
            for Max is returned.
        """
        # check if the state is already in hash table
        if key in self._visited and depth <= self._visited[key].depth:
            return self._visited[key].value

        # if there are no more levels to examine or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get the heuristic value
            value = game.evaluation_function(game.current_player_idx, self._enhance)
            # save the state in hash table
            self._visited[key] = EntryMinMax(0, value)
            # return its heuristic value
            return value

        # set the current best max value
        best_value = float('-inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # for each possible game transition
        for _, state, key in transitions:
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
                # save the state in hash table
                self._visited[key] = EntryMinMax(depth, best_value)
                # terminate the search
                return best_value

        # save the state in hash table
        self._visited[key] = EntryMinMax(depth, best_value)
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
            depth: the remaining depth in the search tree;
            alpha: the best value among all Max ancestors;
            beta: the best value among all Min ancestors.

        Returns:
            The evaluation function value of the best move to play
            for Min is returned.
        """
        # check if the state is already in hash table
        if key in self._visited and depth <= self._visited[key].depth:
            return self._visited[key].value

        # if there are no more levels to examine or we are in a terminal state
        if depth <= 0 or game.check_winner() != -1:
            # get the heuristic value
            value = game.evaluation_function(1 - game.current_player_idx, self._enhance)
            # save the state in hash table
            self._visited[key] = EntryMinMax(0, value)
            # return its heuristic value
            return value

        # set the current best min value
        best_value = float('inf')
        # get all possible game transitions or canonical transitions
        transitions = (
            game.generate_canonical_transitions() if self._symmetries else game.generate_possible_transitions()
        )
        # for each possible game transition
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
                # save the state in hash table
                self._visited[key] = EntryMinMax(depth, best_value)
                # terminate the search
                return best_value

        # save the state in hash table
        self._visited[key] = EntryMinMax(depth, best_value)
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
        transitions = game.generate_canonical_transitions()
        # if we are parallelizing
        if self._parallelize:
            # parallelize min_value
            values = Parallel(n_jobs=-1)(
                delayed(self.min_value)(state, key, self._depth - 1, float('-inf'), float('inf'))
                for _, state, key in transitions
            )
        # otherwise
        else:
            # do not parallelize
            values = [
                self.min_value(state, key, self._depth - 1, float('-inf'), float('inf'))
                for _, state, key in transitions
            ]

        # return the action corresponding to the best estimated move
        _, (action, _, _) = max(
            enumerate(transitions),
            key=lambda t: values[t[0]],
        )
        # return it
        return action


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def show_minmax_statistics(player: MinMaxPlayer) -> None:
        """
        Play a few games between two players and plot the calculated winning percentages.

        Args:
            players1: the first player;
            players2: the second player;
            n_games: how many games to play.

        Returns:
            None.
        """
        import time

        # define the width and height of the figure in inches
        plt.figure(figsize=(8, 5))

        games_duration = []
        flags = []

        # let the different flags
        for flag1, flag2 in [(False,True),(True,True),(False,False),(True,False),]:
            # set flags
            player.parallelize(flag1)
            player._symmetries = flag2
            # create the game
            game = Game()
            # take a start time
            start = time.time()
            # play the game
            game.play(player, RandomPlayer())
            # append total time
            games_duration.append(round(time.time() - start,2))
            # append flags used
            flags.append(f"{'Parallization' if flag1 and not flag2 else ('Parallization and Symmetries' if flag1 and flag2 else ('Symmetries' if flag2 and not flag1 else 'Nothing'))}")
            
        # plot the first player wins
        bars = plt.bar(flags, games_duration, color=['royalblue'], width=0.2)
        # for each bar
        for i,bar in enumerate(bars):
            # write the percentage on top of the bar
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                games_duration[i],
                ha='center',
                va='bottom',
                fontsize='medium',
            )
        
        # delete y-axis ticks and labels
        plt.tick_params(left=False, labelleft=False)
        # specify the y-axis label
        plt.ylabel('Seconds')
        # specify the title shared between the subplots
        plt.title(
            f'Different time of 1 game duration: {player.__class__.__name__} vs RandomPlayer.',
            fontsize=10,
        )
        plt.show()
    minmax_player = AlphaBetaMinMaxPlayer(depth=2, enhance=True)
    show_minmax_statistics(minmax_player)
