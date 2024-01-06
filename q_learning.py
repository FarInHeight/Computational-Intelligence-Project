from game import Game, Move, Player
from typing import Literal
import numpy as np
import pickle
from random import randint, random
from tqdm import trange
from random_player import RandomPlayer


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
        opponent: 'Player',
    ) -> None:
        """
        The Q-learning player constructor.

        Args:
            n_episodes: the number of episodes for the training phase;
            alpha: how much information to incorporate from the new experience;
            gamma: the discount rate of the Bellman equation;
            min_exploration_rate: the minimum rate for exploration during the training phase;
            exploration_decay_rate: the exploration decay rate used during the training;
            opponent: the opponent to play against.

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
        self._opponent = opponent  # define the opponent to play against

    def _move_reward(self, game: 'Game', move: tuple[int, int], player_id: int) -> tuple[Literal[-1, 1], bool]:
        """
        Try a move and return the corresponding reward.

        Args:
            game: a game instance;
            move: the move to try;
            player_id: my player's id.

        Returns:
            The reward and the acceptability of the move are returned.
        """
        # play a move
        acceptable = game.move(move, player_id)
        # give a negative reward to the agent
        reward = -1
        # if the move is acceptable
        if acceptable:
            # give a positive reward to the agent
            reward = 1
        return reward, acceptable

    def _game_reward(self, player: 'Game', winner: int) -> Literal[-10, 0, 10]:
        """
        Calculate the reward based on how the game ended.

        Args:
            player: the winning player;
            winner: the winner's player id.

        Returns:
            The game reward is returned.
        """
        # if there was no winner
        if winner == -1:
            # return no reward
            return 0
        # if the agent is the winner
        elif self == player:
            # give a big positive reward
            return 10
        # give a big negative reward, otherwise
        return -10

    def _map_state_to_index(self, game: 'Game') -> str:
        """
        Given a game state, this function translates it into an index to access the Q_table.

        Args:
            game: a game instance.
        """
        # take the current game state
        state = game.board
        # change not taken tiles values to 2
        state[state == -1] = 2
        # map the state to a string in base 3
        state_repr_index = ''.join(str(_) for _ in state.flatten())
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
            self._q_table[state_repr_index] = np.zeros((9,))
        # if the next state is unknown
        if new_state_repr_index not in self._q_table:
            # create its entry in the action-value mapping table
            self._q_table[new_state_repr_index] = np.zeros((9,))
        prev_value = self._q_table[state_repr_index][action]
        # update the action-value mapping entry for the current state using Q-learning
        self._q_table[state_repr_index][action] = (1 - self._alpha) * prev_value + self._alpha * (
            reward + self._gamma * (-np.max(self._q_table[new_state_repr_index]))
        )

    def _make_move(self, game: 'Game') -> tuple[int, int]:
        """
        Construct a move during the training phase to update the Q_table.

        Args:
            game: a game instance.

        Returns:
            A move to play is returned.
        """
        # get the current state representation
        state_repr_index = self._map_state_to_index(game)

        # randomly perform exploration
        if random() < self._exploration_rate:
            # by returning a random move
            move = randint(0, 8)
        # perform eploitation, otherwise
        else:
            # if the current state is unknown
            if state_repr_index not in self._q_table:
                # create its entry in the action-value mapping table
                self._q_table[state_repr_index] = np.zeros((9,))
            # take the action with maximum return of rewards
            move = np.argmax(self._q_table[state_repr_index])

        # reshape the move to match the board shape
        move = move // 3, move % 3

        return move

    def make_move(self, game: 'Game') -> tuple[int, int]:
        """
        Construct a move to be played according to the Q_table.

        Args:
            game: a game instance.

        Returns:
            A move to play is returned.
        """
        # get the current state representation
        state_repr_index = self._map_state_to_index(game)
        # if the current state is known
        if state_repr_index in self._q_table:
            # take the action with maximum return of rewards
            move = np.argmax(self._q_table[state_repr_index])
            # reshape the move to match the board shape
            move = move // 3, move % 3
            # if the move is acceptable
            if game.is_acceptable(move):
                # return it
                return move
        # perform a random move, otherwise
        return (randint(0, game.board.shape[0] - 1), randint(0, game.board.shape[1] - 1))

    def train(self) -> None:
        """
        Train the Q-learning player.

        Args:
            None.

        Returns:
            None.
        """
        # define the history of rewards
        all_rewards = []
        # define how many episodes to run
        pbar = trange(self._n_episodes)
        # define the players
        players = (self, self._opponent)
        # for each episode
        for episode in pbar:
            # define a new game
            game = Game()
            # sets the rewards to zero
            rewards = 0

            # define a variable to indicate if there is a winner
            winner = -1
            # change players order
            players = (players[1], players[0])
            # define the current player index
            player_idx = 1

            # if we can still play
            while winner < 0 and game.is_still_playable():
                # change player
                player_idx = (player_idx + 1) % 2
                player = players[player_idx]

                # define a variable to check if the chosen move is ok or not
                ok = False
                # if it is our turn
                if self == player:
                    # while the chosen move is not ok
                    while not ok:
                        # get the current state representation
                        state_repr_index = self._map_state_to_index(game)
                        # get a move
                        move = self._make_move(game)
                        # reshape the move to form an index
                        action = move[0] * 3 + move[1]
                        # perform the move and get the reward
                        reward, ok = self._move_reward(game, move, player_idx)
                        # get the next state representation
                        new_state_repr_index = self._map_state_to_index(game)

                        # update the action-value function
                        self._update_q_table(state_repr_index, new_state_repr_index, action, reward)

                        # update the rewards
                        rewards += reward
                # if it is the opponent turn
                else:
                    # while the chosen move is not ok
                    while not ok:
                        # get a move
                        move = player.make_move(game)
                        # perform the move
                        ok = game.move(move, player_idx)

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
            # update the rewards
            rewards += reward
            # update the rewards history
            all_rewards.append(rewards)
            pbar.set_description(f'rewards value: {rewards}, current exploration rate: {self._exploration_rate:2f}')

        print(f'** Last 1_000 episodes - Mean rewards value: {sum(all_rewards[-1_000:]) / 1_000:.2f} **')
        print(f'** Last rewards value: {all_rewards[-1]:} **')

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
        n_episodes=500_000,
        alpha=0.1,
        gamma=0.99,
        min_exploration_rate=0.01,
        exploration_decay_rate=3e-6,
        opponent=RandomPlayer(),
    )
    # train the Q-learning player
    q_learning_rl_agent.train()
    # print the number of explored states
    print(f'Number of explored states: {len(q_learning_rl_agent._q_table.keys())}')
    # serialize the Q-learning player
    q_learning_rl_agent.save()
