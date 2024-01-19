# Computational Intelligence Project

## Players - Design Choices

Since during the semester we developed several agents based on the techniques explained in the lectures, we mainly focused our project on methods which we did not develop in the laboratories or in the additional material we proposed in our personal repositories.

Keeping this in mind, we decided to implement the following methods:
- [x] Human Player
- [x] MinMax
- [x] MinMax + Alpha-Beta pruning
- [x] Monte Carlo Reinforcement Learning
- [x] Monte Carlo Tree Search

Although _Monte Carlo Tree Search_ is not a topic of the course, we included it because _Quixo_ has a great branching factor value and we wanted an agent that could overcome this problem.

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

## Resources

* Sutton & Barto, _Reinforcement Learning: An Introduction_ [2nd Edition]
* Russel, Norvig, _Artificial Intelligence: A Modern Approach_ [4th edition]
* Nils J. Nilsson, _Artificial Intelligence: A New Synthesis_ Morgan Kaufmann Publishers, Inc. (1998)
* [aimacode/aima-python](https://github.com/aimacode/aima-python/tree/master) + [Monte Carlo Tree Search implementation example](https://github.com/aimacode/aima-python/blob/master/games4e.py#L178)

## License
[MIT License](LICENSE)