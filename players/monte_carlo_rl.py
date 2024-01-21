from game import Game, Move, Player
from typing import Literal
import numpy as np
import pickle
import math
from random import random, choice
from tqdm import trange
from players.random_player import RandomPlayer
from utils.investigate_game import InvestigateGame, MissNoAddDict
from players.min_max import AlphaBetaMinMaxPlayer


class MonteCarloRLPlayer(Player):
    """
    Class representing a player who learns to play thanks to the Monte Carlo-learning technique.
    """

    def __init__(
        self,
        n_episodes: int = 500_000,
        gamma: float = 0.95,
        alpha: float = 0.1,
        min_exploration_rate: float = 0.01,
        exploration_decay_rate: float = 1e-5,
    ) -> None:
        """
        The Monte Carlo-learning player constructor.

        Args:
            n_episodes: the number of episodes for the training phase;
            gamma: the discount rate;
            alpha: how much information to incorporate from the new experience;
            min_exploration_rate: the minimum rate for exploration during the training phase;
            exploration_decay_rate: the exploration decay rate used during the training;

        Returns:
            None.
        """
        super().__init__()
        self._state_values = MissNoAddDict(float)  # define the State-Value function
        self._n_episodes = n_episodes  # define the number of episodes for the training phase
        self._gamma = gamma  # define the discount rate
        self._alpha = alpha  # define how much information to incorporate from the new experience
        self._exploration_rate = 1  # define the exploration rate for the training phase
        self._min_exploration_rate = (
            min_exploration_rate  # define the minimum rate for exploration during the training phase
        )
        self._exploration_decay_rate = (
            exploration_decay_rate  # define the exploration decay rate used during the training
        )
        self._rewards = []  # list of the rewards obtained during training

    @property
    def rewards(self) -> list[int]:
        """
        Return a copy of the rewards obtained during training

        Args:
            None.

        Returns:
            The training rewards are returned.
        """
        return tuple(self._rewards)

    def _game_reward(self, player: 'InvestigateGame', winner: int) -> Literal[-10, -1, 10]:
        """
        Calculate the reward based on how the game ended.

        Args:
            player: the winning player;
            winner: the winner's player id.

        Returns:
            The game reward is returned.
        """
        # if no one wins
        if winner == -1:
            # return a small penalty
            return -1
        # if the agent is the winner
        if self == player:
            # give a big positive reward
            return 10
        # give a big negative reward, otherwise
        return -10

    def _update_state_values(self, trajectory: list, reward: float) -> None:
        """
        Update the State-Value function according to the Monte Carlo-learning technique.

        Args:
            trajectory: the trajectory of the current episode;
            reward: the final reward of the game.

        Returns:
            None.
        """
        # define the return of rewards
        return_of_rewards = reward
        # for each state in the trajectory
        for state_repr_index in reversed(trajectory):
            # update the State-Value mapping table
            self._state_values[state_repr_index] = self._state_values[state_repr_index] + self._alpha * (
                self._gamma * return_of_rewards - self._state_values[state_repr_index]
            )
            return_of_rewards = self._state_values[state_repr_index]

    def _step_training(
        self,
        game: 'InvestigateGame',
        player_id: int,
    ) -> tuple[tuple[tuple[int, int], Move], 'InvestigateGame']:
        """
        Construct a move during the training phase to update the State-Value function.

        Args:
            game: a game instance;
            player_id: my player's id.

        Returns:
            A move to play is returned.
        """

        # get all possible canonical transitions
        transitions = game.generate_canonical_transitions(player_id)

        # randomly perform exploration
        if random() < self._exploration_rate:
            # choose a random transition
            transition = choice(transitions)
        # perform eploitation, otherwise
        else:
            # take the action with maximum return of rewards
            transition = max(transitions, key=lambda t: self._state_values[t[2]])

        return transition

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Construct a move to be played according to the State-Value function.

        Args:
            game: a game instance.

        Returns:
            A move to play is returned.
        """
        # create seperate instance of a game for investigation
        game = InvestigateGame(game)
        # get all possible canonical transitions
        transitions = game.generate_canonical_transitions()
        # take the action with maximum return of rewards
        action, _, _ = max(transitions, key=lambda t: self._state_values[t[2]])

        # return the action
        return action

    def train(
        self,
        max_steps_draw: int,
        opponent: Player = AlphaBetaMinMaxPlayer(depth=1),
        switch_ratio: int = 1,
    ) -> None:
        """
        Train the Monte Carlo-learning player.

        Args:
            max_steps_draw: define the maximum number of steps before
                            claiming a draw.
            opponent: the other player against which we play after the switch moment;
            switch_ratio: define the moment in which we should play against the other opponent.

        Returns:
            None.
        """

        # define how many episodes to run
        pbar_episodes = trange(self._n_episodes)
        # define the players
        players = self, RandomPlayer()

        # if we want to play also against minmax
        if self._minmax:
            # define a new players tuple
            new_players = self, opponent

        # for each episode
        for episode in pbar_episodes:
            # define a new game
            game = InvestigateGame(Game())

            # switch the players if it is the moment
            if math.isclose(switch_ratio, episode / self._n_episodes):
                players = new_players

            # define the trajectory
            trajectory = []

            # define a variable to indicate if there is a winner
            winner = -1
            # change players order
            players = players[1], players[0]
            # define the current player index
            player_idx = 1

            # save last action
            last_action = None
            # define counter to terminate if we are in a loop
            counter = 0

            # if we can still play
            while winner < 0 and counter < max_steps_draw:
                # change player
                player_idx = (player_idx + 1) % 2
                player = players[player_idx]

                # if it is our turn
                if self == player:
                    # get an action
                    action, game, state_repr_index = self._step_training(game, player_idx)

                    # update the trajectory
                    trajectory.append(state_repr_index)

                    # if we play the same action as before
                    if last_action == action:
                        # increment the counter
                        counter += 1
                    # otherwise
                    else:
                        # save the new last action
                        last_action = action
                        # reset the counter
                        counter = 0

                # if it is the opponent's turn
                else:
                    # define a variable to check if the chosen move is ok or not
                    ok = False
                    # while the chosen move is not ok
                    while not ok:
                        # get a move
                        move = player.make_move(game)
                        # perform the move
                        ok = game._Game__move(*move, player_idx)
                    # update the trajectory
                    # trajectory.append(Symmetry.get_canonical_state(game, 1 - player_idx))

                # check if there is a winner
                winner = game.check_winner()

            # update the exploration rate
            self._exploration_rate = np.clip(
                np.exp(-self._exploration_decay_rate * episode), self._min_exploration_rate, 1
            )

            # get the game reward
            reward = self._game_reward(player, winner)
            # update the rewards history
            self._rewards.append(reward)
            # update the State-Value function
            self._update_state_values(trajectory, reward)

            pbar_episodes.set_description(
                f"# last 1_000 episodes mean reward: {sum(self._rewards[-1_000:]) / 1_000:.2f} - # explored states: {len(self._state_values):,} - Current exploration rate: {self._exploration_rate:2f}"
            )

        print(f'** Last 1_000 episodes - Mean rewards value: {sum(self._rewards[-1_000:]) / 1_000:.2f} **')
        print(f'** Last rewards value: {self._rewards[-1]:} **')

    def save(self, path: str) -> None:
        """
        Serialize the current Monte Carlo learning player's state.

        Args:
            path: location where to save the player's state.

        Returns: None.
        """
        # serialize the Monte Carlo learning player
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """
        Load a Monte Carlo learning player's state into the current player.

        Args:
            path: location from which to load the player's state.

        Returns: None.
        """
        # load the serialized Monte Carlo learning player
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)


if __name__ == '__main__':
    # create the player
    monte_carlo_rl_agent = MonteCarloRLPlayer()
    # train the player
    monte_carlo_rl_agent.train(max_steps_draw=10)
    # serialize the player
    monte_carlo_rl_agent.save('agents/monte_carlo_rl_agent_no_sim2.pkl')
