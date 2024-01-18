import numpy as np
from game import Game, Move, Player
from utils.investigate_game import InvestigateGame
from players.random_player import RandomPlayer
from random import choice


class NodeMCT:
    """
    Class representing a node for the Monte Carlo Tree Search algorithm.
    """

    def __init__(self, state: 'InvestigateGame', parent: 'NodeMCT' = None):
        """
        The Monte Carlo Tree Search Node constructor.

        Args:
            state: the game instance to store in the node;
            parent: the parent node of this node.

        Returns:
            None.

        """
        self.state = state  # the game instance for this node
        self.parent = parent  # the node's parent
        self.utility = 0  # the utility of the node
        self.n_games = 0  # number of simulations played for this node
        self.children = None  # a dictionary that links an action with a child

    def is_terminal(self) -> bool:
        """
        Function that check if the node is a terminal node.

        Args:
            None.

        Returns:
            If the state is terminal or not.
        """
        return self.state.check_winner() != -1


class MCTSPlayer(Player):
    """
    Class representing the Monte Carlo Tree Search Player.
    """

    def __init__(self, n_simulations: int = 300, symmetries: bool = False, random: bool = False) -> None:
        """
        The Monte Carlo Tree Search Player constructor.

        Args:
            n_simulations: the number of simulations for each move;
            symmetries: flag to decide if apply symmetries or not;
            random: flag to decide if play random simulations or not.

        Returns:
            None.
        """
        self._n_simulations = n_simulations  # the number of simulations to play
        self._symmetries = symmetries  # do we want to use symmetries?
        self._random = random  # do we want to play randomly during a simulation?
        self._random_player = RandomPlayer()  # instantiate the random player

    @classmethod
    def ucb(cls, node: NodeMCT, C=1.4) -> float:
        """
        Method to compute the Upper-Confidence Bound function.

        Args:
            node: the node on which to calculate the UCB;
            C: the constant factor to use in the function.

        Returns:
            The UCB value of the node.
        """
        return (
            np.inf
            if node.n_games == 0
            else node.utility / node.n_games + C * np.sqrt(np.log(node.parent.n_games) / node.n_games)
        )

    def _select(self, node: NodeMCT) -> NodeMCT:
        """
        Function that select a node based on the UCB value.

        Args:
            node: the parent node;

        Returns:
            The selected node.
        """
        # if the node has children
        if node.children:
            # select the best descendant according to the UCB value
            return self._select(max(node.children.values(), key=MCTSPlayer.ucb))
        # otherwise
        else:
            # return the node itself
            return node

    def _expand(self, node: NodeMCT) -> NodeMCT:
        """
        Function that expand a non terminal node.

        Args:
            node: the node that is to be expanded.

        Returns:
            Return a node among the children.
        """
        # if the current node has no children and it is not terminal
        if not node.children and not node.is_terminal():
            # compute the possible transitions of the current game state
            transitions = (
                node.state.generate_canonical_transitions()
                if self._symmetries
                else node.state.generate_possible_transitions()
            )
            # create the children
            node.children = {action: NodeMCT(state=next_state, parent=node) for action, next_state, _ in transitions}

        # return a new node among the children
        return self._select(node)

    def _simulate(self, state: InvestigateGame) -> int:
        """
        Function that simulates a game.

        Args:
            state: an instance of the Game;

        Returns:
            The utility of the simulation is returned.
        """
        # get my id
        player_id = state.get_current_player()

        # set a counter to determine whether or not to end the game
        count = 0
        # define a variable to save the last played action
        last_action = None
        # while we can still play
        while state.check_winner() == -1 and count < 5:
            # if we have to play random
            if self._random:
                # define a variable to check if the chosen move is ok or not
                ok = False
                # while the chosen move is not ok
                while not ok:
                    # choose a random move
                    action = self._random_player.make_move(state)
                    # check if it is valid
                    ok = state._Game__move(*action, state.get_current_player())
            # otherwise
            else:
                # compute the possible transitions of the current game state
                transitions = (
                    state.generate_canonical_transitions()
                    if self._symmetries
                    else state.generate_possible_transitions()
                )
                # sort them according to the evaluation function
                transitions = sorted(transitions, key=lambda x: x[1].evaluation_function(player_id))
                # choose a random move among the top 3 actions
                action, state, _ = choice(transitions[-3:])
            # if I am the player who has to move
            if player_id == state.get_current_player():
                # if we play the same action as before
                if last_action == action:
                    # update the counter
                    count += 1
                # otherwise
                else:
                    # reset the counter
                    count = 0
                    # save the new last action
                    last_action = action

        # if I am the winner
        if player_id == state.check_winner():
            # return a negative reward to my parent
            return -1

        # return a positive reward to my parent
        return 1

    def _backpropagate(self, node: NodeMCT, utility: float) -> None:
        """
        Recursive function that backpropagate the utility in the Tree.

        Args:
            node: the current node to update;
            utility: the current value of utility;

        Return:
            None.
        """
        # it the utility is positive
        if utility == 1:
            # update the node's utility
            node.utility += utility

        # update the node's number of games
        node.n_games += 1

        # if the node is not the root node
        if node.parent:
            # backpropagate the opposite utility
            self._backpropagate(node.parent, -utility)

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        """
        Construct a move to be played according to the results of the simulations.

        Args:
            game: a game instance.

        Returns:
            A move to play is returned.
        """
        # create a variable to investigate the game
        game = InvestigateGame(game)
        # create the root node
        root = NodeMCT(game)
        # for each simulation
        for _ in range(self._n_simulations):
            # select a lead node
            leaf = self._select(root)
            # selecting a newborn child
            child = self._expand(leaf)
            # simulate the game starting for this newborn child
            result = self._simulate(child.state)
            # backpropagate the result
            self._backpropagate(child, result)

        # select the best action to play
        best_action = max(root.children.items(), key=lambda child: child[1].n_games)[0]

        return best_action
