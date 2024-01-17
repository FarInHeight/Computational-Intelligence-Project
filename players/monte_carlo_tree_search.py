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
            state: the game instance of the node;
            parent: the parent node of the Tree.

        Returns:
            None.

        """
        self.state = state
        self.parent = parent
        self.utility = 0  # the utility of the node
        self.n_games = 0  # number of simulations played starting from the node
        self.children = None  # a dictionary that links every action with the next node

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
            symmetries: flag to decide if apply symmetries detection or not;
            random: flag to decide if play random simulations.

        Returns:
            None.
        """
        self._n_simulations = n_simulations
        self._symmetries = symmetries
        self._random = random
        self._calls = 0
        self._random_player = RandomPlayer()

    @classmethod
    def ucb(cls, node: NodeMCT, C=1.4) -> float:
        """
        Method to compute the Upper-Confidence Bound function.

        Args:
            node: the node on which to calculate the ucb;
            C: a factor

        Returns:
            The ucb value of the node.
        """
        return (
            np.inf
            if node.n_games == 0
            else node.utility / node.n_games + C * np.sqrt(np.log(node.parent.n_games) / node.n_games)
        )

    def _select(self, node: NodeMCT) -> NodeMCT:
        """
        Function that select a node based on ucb value.

        Args:
            node: the parent node;

        Returns:
            The selected node.
        """
        if node.children:
            return self._select(max(node.children.values(), key=MCTSPlayer.ucb))
        else:
            return node

    def _expand(self, node: NodeMCT) -> NodeMCT:
        """
        Function that expand a non terminal node.

        Args:
            node: the node that is to be expanded.

        Returns:
            Return a node among the children.
        """
        if not node.children and not node.is_terminal():
            transitions = (
                node.state.generate_canonical_transitions()
                if self._symmetries
                else node.state.generate_possible_transitions()
            )
            node.children = {action: NodeMCT(state=next_state, parent=node) for action, next_state, _ in transitions}

        return self._select(node)

    def _simulate(self, state: InvestigateGame) -> int:
        """
        Function that simulate a game.

        Args:
            state: an instance of the Game;

        Returns:
            The utility of the simulation is turned.
        """
        player_id = state.get_current_player()

        count = 0
        last_action = None
        while state.check_winner() == -1 and count < 5:
            if self._random:
                ok = False
                while not ok:
                    action = self._random_player.make_move(state)
                    ok = state._Game__move(*action, state.get_current_player())
            else:
                transitions = (
                    state.generate_canonical_transitions()
                    if self._symmetries
                    else state.generate_possible_transitions()
                )
                self._calls += 1
                transitions = sorted(transitions, key=lambda x: x[1].evaluation_function(player_id))
                action, state, _ = choice(transitions[:-3])
            if player_id == state.get_current_player():
                if last_action == action:
                    count += 1
                else:
                    count = 0
                    last_action = action

        if player_id == state.check_winner():
            return -1

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
        if utility == 1:
            node.utility += utility
        node.n_games += 1

        if node.parent:
            self._backpropagate(node.parent, -utility)

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        """
        Construct a move to be played according to results of simulations.

        Args:
            game: a game instance.

        Returns:
            A move to play is returned.
        """
        game = InvestigateGame(game)
        root = NodeMCT(game)
        for _ in range(self._n_simulations):
            leaf = self._select(root)
            child = self._expand(leaf)
            result = self._simulate(child.state)
            self._backpropagate(child, result)

        best_action = max(root.children.items(), key=lambda child: child[1].n_games)[0]

        return best_action
