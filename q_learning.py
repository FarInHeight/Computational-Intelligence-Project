from game import Game, Move, Player
from typing import Literal
import numpy as np
import pickle
import math
from random import random, choice
from tqdm import trange
from random_player import RandomPlayer
from collections import defaultdict
from investigate_game import InvestigateGame
from min_max import MinMaxPlayer


class QLearningRLPlayer(Player):
    """
    Class representing player who learns to play thanks to the Q-learning technique.
    """

    def __init__(
        self,
        n_episodes: int,
        alpha: float,
        gamma: float,
        min_exploration_rate: float,
        exploration_decay_rate: float,
        minmax: bool = False,
        switch_ratio: int = 0.8,
        depth: int = 1,
        symmetries: bool = False,
    ) -> None:
        """
        The Q-learning player constructor.

        Args:
            n_episodes: the number of episodes for the training phase;
            alpha: how much information to incorporate from the new experience;
            gamma: the discount rate of the Bellman equation;
            min_exploration_rate: the minimum rate for exploration during the training phase;
            exploration_decay_rate: the exploration decay rate used during the training;
            minmax: decide if the training must be performed also on minmax.
            switch_ratio: define the moment in which we should play against minmax;
            depth: maximum depth of the Min-Max search tree;
            symmetries: flag to consider the symmetries or not.

        Returns:
            None.
        """
        super().__init__()
        self._q_table = {}  # define the Action-value function
        self._n_episodes = n_episodes  # define the number of episodes for the training phase
        self._alpha = alpha  # define how much information to incorporate from the new experience
        self._gamma = gamma  # define the discount rate of the Bellman equation
        self._exploration_rate = 1  # define the exploration rate for the training phase
        self._min_exploration_rate = (
            min_exploration_rate  # define the minimum rate for exploration during the training phase
        )
        self._exploration_decay_rate = (
            exploration_decay_rate  # define the exploration decay rate used during the training
        )
        self._minmax = minmax  # define if we want to play also against minmax
        self._switch_ratio = switch_ratio  # define the moment in which minmax plays against us
        self._depth = depth  # define the depth for minmax
        self._symmetries = symmetries  # choose if play symmetries should be considered
        self._rewards = []  # list of the rewards obtained during training

    @property
    def rewards(self) -> list[int]:
        """
        Return a copy the the rewards obtained during training

        Args:
            None.

        Returns:
            The training rewards are returned.
        """
        return tuple(self._rewards)

    def _game_reward(self, player: 'InvestigateGame', winner: int) -> Literal[-10, 0, 10]:
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
            # return small penalty
            return -1
        # if the agent is the winner
        if self == player:
            # give a big positive reward
            return 10
        # give a big negative reward, otherwise
        return -10

    def _map_state_to_index(self, game: 'Game', player_id: int) -> str:
        """
        Given a game state, this function translates it into an index to access the Q_table.

        Args:
            game: a game instance;
            player_id: my player's id.
        """
        # take the current game state
        state = game.get_board()
        # change not taken tiles values to 0
        state += 1
        # map the state to a string in base 3
        state_repr_index = ''.join(str(_) for _ in state.flatten()) + str(player_id)
        return state_repr_index

    def _update_q_table(self, state_repr_index: str, new_state_repr_index: str, action: int, reward: int) -> None:
        """
        Update the Q_table according to the Q-learning update formula.

        Args:
            state_repr_index: the current state index;
            new_state_repr_index: the next state index;
            action: the performed action;
            reward: the reward obtained by applying the action on the current state.

        Returns:
            None.
        """
        # if the current state is unknown
        if state_repr_index not in self._q_table:
            # create its entry in the action-value mapping table
            self._q_table[state_repr_index] = defaultdict(float)
        # if the next state is unknown
        if new_state_repr_index not in self._q_table:
            # create its entry in the action-value mapping table
            self._q_table[new_state_repr_index] = defaultdict(float)
        prev_value = self._q_table[state_repr_index][action]
        # update the action-value mapping entry for the current state using Q-learning
        self._q_table[state_repr_index][action] = (1 - self._alpha) * prev_value + self._alpha * (
            reward + self._gamma * (-max(self._q_table[new_state_repr_index].values(), default=0.0))
        )

    def _step_training(
        self, game: 'InvestigateGame', player_id: int
    ) -> tuple[tuple[tuple[int, int], Move], 'InvestigateGame']:
        """
        Construct a move during the training phase to update the Q_table.

        Args:
            game: a game instance;
            player_id: my player's id.

        Returns:
            A move to play is returned.
        """
        # get the current state representation
        state_repr_index = self._map_state_to_index(game, player_id)
        # get all possible transitions
        transitions = game.generate_possible_transitions(player_id)

        # randomly perform exploration
        if random() < self._exploration_rate:
            # choose a random transition
            transition = choice(transitions)
        # perform eploitation, otherwise
        else:
            # if the current state is unknown
            if state_repr_index not in self._q_table:
                # create its entry in the action-value mapping table
                self._q_table[state_repr_index] = defaultdict(float)
                # choose a random transition
                transition = choice(transitions)
            else:
                # take the action with maximum return of rewards
                transition = max(transitions, key=lambda t: self._q_table[state_repr_index][t[0]])

        return transition

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Construct a move to be played according to the Q_table.

        Args:
            game: a game instance.

        Returns:
            A move to play is returned.
        """
        # create seperate instance of a game for investigation
        game = InvestigateGame(game)
        # get my id
        player_id = game.get_current_player()
        # get the current state representation
        state_repr_index = self._map_state_to_index(game, player_id)
        # get all possible transitions
        actions, _ = zip(*game.generate_possible_transitions(player_id))
        # if the current state is known
        if state_repr_index in self._q_table:
            # take the action with maximum return of rewards
            action = max(actions, key=lambda a: self._q_table[state_repr_index][a])
        else:
            # choose a random action
            action = choice(actions)

        # return the action
        return action

    def train(self, max_steps_draw: int) -> None:
        """
        Train the Q-learning player.

        Args:
            max_steps_draw: define the maximum number of steps before
                            claiming a draw.

        Returns:
            None.
        """
        # save last action
        last_action = None

        # define how many episodes to run
        pbar_episodes = trange(self._n_episodes)

        # define the random tuples
        player_tuples = ((RandomPlayer(), self), (self, RandomPlayer()))

        # if we want to play also against minmax
        if self._minmax:
            # define the minmax players
            minmax_players = (
                (MinMaxPlayer(player_id=0, depth=self._depth, symmetries=self._symmetries), self),
                (self, MinMaxPlayer(player_id=1, depth=self._depth, symmetries=self._symmetries)),
            )

        # for each episode
        for episode in pbar_episodes:
            # define a new game
            game = InvestigateGame(Game())
            # define a variable to indicate if there is a winner

            # switch the players if it is the moment
            if self._minmax and math.isclose(self._switch_ratio, episode / self._n_episodes):
                player_tuples = minmax_players

            winner = -1
            # change player tuple order
            player_tuples = (player_tuples[1], player_tuples[0])
            # change players order
            players = player_tuples[-1]
            # define the current player index
            player_idx = 1

            # define counter to terminate if we are in a loop
            counter = 0

            # if we can still play
            while winner < 0 and counter < max_steps_draw:
                # change player
                player_idx = (player_idx + 1) % 2
                player = players[player_idx]

                # if it is our turn
                if self == player:
                    # get the current state representation
                    state_repr_index = self._map_state_to_index(game, player_idx)
                    # get an action
                    action, game = self._step_training(game, player_idx)
                    # get the next state representation
                    new_state_repr_index = self._map_state_to_index(game, player_idx)
                    # update the action-value function
                    self._update_q_table(state_repr_index, new_state_repr_index, action, reward=0)

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

                # if it is the opponent turn
                else:
                    # define a variable to check if the chosen move is ok or not
                    ok = False
                    # while the chosen move is not ok
                    while not ok:
                        # get a move
                        move = player.make_move(game)
                        # perform the move
                        ok = game._Game__move(*move, player_idx)

                # check if there is a winner
                winner = game.check_winner()

            # update the exploration rate
            self._exploration_rate = np.clip(
                np.exp(-self._exploration_decay_rate * episode), self._min_exploration_rate, 1
            )
            # get the game reward
            reward = self._game_reward(player, winner)
            # update the action-value function
            self._update_q_table(state_repr_index, new_state_repr_index, action, reward)

            # update the rewards history
            self._rewards.append(reward)
            pbar_episodes.set_description(
                f"Win? {'Yes' if reward == 10 else ('Draw' if reward == -1 else 'No') } - Current exploration rate: {self._exploration_rate:2f}"
            )

        print(f'** Last 1_000 episodes - Mean rewards value: {sum(self._rewards[-1_000:]) / 1_000:.2f} **')
        print(f'** Last rewards value: {self._rewards[-1]:} **')

    def save(self, path: str) -> None:
        """
        Serialize the current Q-learning player's state.

        Args:
            path: location where to save the player's state.

        Returns: None.
        """
        # serialize the Q-learning player
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """
        Load a Q-learning player's state into the current player.

        Args:
            path: location from which to load the player's state.

        Returns: None.
        """
        # load the serialized Q-learning player
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)


if __name__ == '__main__':
    # create the Q-learning player
    q_learning_rl_agent = QLearningRLPlayer(
        n_episodes=10_000,
        alpha=0.1,
        gamma=0.99,
        min_exploration_rate=0.01,
        exploration_decay_rate=1e-4,
        minmax=True,
    )
    # train the Q-learning player
    q_learning_rl_agent.train(max_steps_draw=10)
    # get the rewards
    rewards = q_learning_rl_agent.rewards
    # print flash statistics
    print(rewards.count(-10), rewards.count(10), rewards.count(-1))
    # print the number of explored states
    print(f'Number of explored states: {len(q_learning_rl_agent._q_table.keys())}')
    # serialize the Q-learning player
    q_learning_rl_agent.save('agents/q_learning_rl_agent.pkl')
