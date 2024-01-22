# Computational Intelligence Project - Quixo

## Game description

**Quixo** is a game based on simple rules, but it is by no means simple.

### [Rules](https://cs.uwaterloo.ca/~dtompkin/dtlib/base/Quixo.pdf)

<blockquote>

In turn, each player chooses a cube and moves it according to the following rules. In no event can a player
miss his/her turn. \
### Choosing and taking a cube
The player chooses and takes a blank cube, or one with his/her symbol on it,
from the board’s periphery. In the first round, the players have no choice but to take a blank cube. You
are not allowed to take a cube bearing your opponent’s symbol.
### Changing the cube symbol
Whether the cube taken is blank or already bears the player’s symbol, it must
always be replaced by a cube with the player’s symbol on the top face.
### Replacing the cube
The player can choose at which end of the incomplete rows made when a cube is taken,
the cube is to be replaced; he/she pushes this end to replace the cube. You can never replace the cube just
played back in the position from which it was taken.
### END OF GAME
The winner is the player to make and announce that he/she has made a horizontal, vertical
or diagonal line with 5 cubes bearing his/her symbol. The player to make a line with his/her opponent’s
symbol loses the game, even if he/she makes a line with his/her own symbol at the same time.

</blockquote>

## Players - Design Choices

Since during the semester we developed several agents based on the techniques explained in the lectures, we mainly focused our project on methods which we did not develop in the laboratories or in the additional material we proposed in our personal repositories.

Keeping this in mind, we decided to implement the following methods:
- [x] Human Player
- [x] MinMax
- [x] MinMax + Alpha-Beta pruning
- [x] Monte Carlo Reinforcement Learning (TD learning + Symmetries)
- [x] Monte Carlo Tree Search

Although _Monte Carlo Tree Search_ is not a topic of the course, we included it because _Quixo_ has a great branching factor value and we wanted an agent that could overcome this problem.

### Space Optimization

Since the _Quixo_ game has a huge amount of states, we focused our attention on optimizing the space required by our serialized agents. Before this new representation, the Monte Carlo RL player weighed more than a GB, while now its size is 57 KB.

### Players Improvements

To improve the performance of the players we implemented the following improvements:
- [x] parallelization
- [x] hash tables
- [x] symmetries

### Failed Attempts

We also tried to include in the project a Q-learning player, but we failed resoundingly due to the huge amount of _state-action_ pairs to learn. For this reason, we removed it from the repository.

We tried to use the same agents implemented for the last laboratory, but we failed because the formulas we used were not sufficient to learn the return of rewards of the millions and millions of states in which _Quixo_ can be found. \
We performed several trials and after a consultation with [Riccardo Cardona](https://github.com/Riden15/Computational-Intelligence), we found that the formula he used for the project is quite efficient and effective.

## Repository Structure

- [players](players): this directory contains the implemented agents
    - [human_player.py](players/human_player.py): class which implements a human player
    - [min_max.py](players/min_max.py): class which implements the MinMax algorithm and the Alpha-Beta pruning technique
    - [monte_carlo_rl.py](players/monte_carlo_rl.py): class which implements the Monte Carlo Reinforcement Learning player
    - [monte_carlo_tree_search.py](players/monte_carlo_tree_search.py): class which implements the Monte Carlo Tree Search algorithm
    - [random_player.py](players/random_player.py): class which implements a player that plays randomly
- [trained_agents](trained_agents): this directory contains the trained agents
- [utils](utils): this directory contains files which are necessary for the agents to play and implement performance improvements
    - [investigate_game.py](utils/investigate_game.py): class which extends `Game` and it is used by our agents 
    - [symmetry.py](utils/symmetry.py): class which implements all the possible symmetries and it is used by our agents
- [project_summary.ipynb](project_summary.ipynb): notebook used to train agents and to show results

The serialized `MinMax` and `MinMax + Alpha-Beta pruning` players with a non-empty hash table can be found in the [release section](https://github.com/FarInHeight/Computational-Intelligence-Project/releases/tag/v1.0.0).

## How to run

To run a specific `module.py` file, open the terminal and type the following command from the root of the project:
```bash
python -m folder.module
```
As an example, run the `min_max.py` file as follows:
```bash
python -m players.min_max
```

If you are using VS Code as editor, you can add 
```json
"terminal.integrated.env.[your os]": 
{
    "PYTHONPATH": "${workspaceFolder}"
}
```
to your settings and run the module directly using the <kbd>▶</kbd> button.

## Resources

* Sutton & Barto, _Reinforcement Learning: An Introduction_ [2nd Edition]
* Russel, Norvig, _Artificial Intelligence: A Modern Approach_ [4th edition]
* Nils J. Nilsson, _Artificial Intelligence: A New Synthesis_ Morgan Kaufmann Publishers, Inc. (1998)
* [Quixo Is Solved](https://arxiv.org/pdf/2007.15895.pdf)
* [aimacode/aima-python](https://github.com/aimacode/aima-python/tree/master) + [Monte Carlo Tree Search implementation example](https://github.com/aimacode/aima-python/blob/master/games4e.py#L178)

## License
[MIT License](LICENSE)