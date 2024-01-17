import numpy as np
import time
from game import Game, Move, Player
from investigate_game import InvestigateGame
from random_player import RandomPlayer
from random import choice


class NodeMCT:
    def __init__(self, state: 'InvestigateGame', parent: 'NodeMCT' = None):
        self.state = state
        self.parent = parent
        self.utility = 0
        self.n_games = 0
        self.children = None

    def is_terminal(self):
        return self.state.check_winner() != -1


class MCTSPlayer(Player):
    def __init__(self, n_simulations: int = 300, symmetries: bool = False, random: bool = False) -> None:
        self._n_simulations = n_simulations
        self._symmetries = symmetries
        self._random = random
        self._calls = 0
        self._random_player = RandomPlayer()

    @classmethod
    def ucb(cls, node: NodeMCT, C=1.4):
        return (
            np.inf
            if node.n_games == 0
            else node.utility / node.n_games + C * np.sqrt(np.log(node.parent.n_games) / node.n_games)
        )

    def _select(self, node: NodeMCT):
        if node.children:
            return self._select(max(node.children.values(), key=MCTSPlayer.ucb))
        else:
            return node

    def _expand(self, node: NodeMCT):
        if not node.children and not node.is_terminal():
            transitions = (
                node.state.generate_canonical_transitions()
                if self._symmetries
                else node.state.generate_possible_transitions()
            )
            self._calls += 1
            node.children = {action: NodeMCT(state=next_state, parent=node) for action, next_state, _ in transitions}

        return self._select(node)

    def _simulate(self, state: InvestigateGame):
        player_id = state.get_current_player()

        max_depth = 50
        while state.check_winner() == -1 and max_depth > 0:
            if self._random:
                ok = False
                while not ok:
                    from_pos, slide = self._random_player.make_move(state)
                    ok = state._Game__move(from_pos, slide, state.get_current_player())
            else:
                transitions = (
                    state.generate_canonical_transitions()
                    if self._symmetries
                    else state.generate_possible_transitions()
                )
                self._calls += 1
                _, state, _ = max(transitions, key=lambda x: x[1].evaluation_function(player_id))
            max_depth -= 1

        if max_depth == 0:
            print('Loop')

        if player_id == state.check_winner():
            return -1

        return 1

    def _backpropagate(self, node: NodeMCT, utility: float):
        if utility == 1:
            node.utility += utility
        node.n_games += 1

        if node.parent:
            self._backpropagate(node.parent, -utility)

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        game = InvestigateGame(game)
        root = NodeMCT(game)
        for _ in range(self._n_simulations):
            leaf = self._select(root)
            child = self._expand(leaf)
            result = self._simulate(child.state)
            self._backpropagate(child, result)

        best_action = max(root.children.items(), key=lambda child: child[1].n_games)[0]

        return best_action


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

    # print(f'MCTSPlayer as first')
    # test(MCTSPlayer(), RandomPlayer(), 3, 0)
    # print(f'MCTSPlayer as second')
    # test(RandomPlayer(), MCTSPlayer(), 3, 1)

    print(f'MCTSPlayer as first')
    test(MCTSPlayer(50, random=False), RandomPlayer(), 3, 0)
    print(f'MCTSPlayer as second')
    test(RandomPlayer(), MCTSPlayer(50, random=False), 3, 1)
    """ player = MCTSPlayer(50, random=False)
    Game().play(player, RandomPlayer())
    print(player._calls)
 """
