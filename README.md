 # knapsack-problem

Comparison of the performance and results of different algorithms used for the knapsack problem.

The problem is a combinatorial optimization problem:
given a backpack with a predetermined capacity, you must find the best combination of objects that can be inserted in such a way as to maximize the number of objects in the backpack without exceeding the total weight. a more detailed explanation
the problem is available [here](https://en.wikipedia.org/wiki/Knapsack_problem).

In particular, I use challenge [Kaggle](https://www.kaggle.com) to compare different approaches to the problem. The challenge is available [here](https://www.kaggle.com/c/santas-uncertain-bags) and consisted of filling the bags of gifts to maximize the number of gifts inthe bags respecting certain constraints.

Two types of algorithms have been used to solve the problem: a *greedy algorithm* and a *genetic algorithm*.

### Definition and constraints of the problem
There are 9 types of gifts associated with a weight probability distribution. In total there are 7166 gifts to be inserted in 1000 bags. Once the weight of each toy has been calculated, through the correct distribution, the combinations of gifts must be found in order to fill the various bags in order to maximize the number of gifts per bag.
There are constraints that must be respected:
- each gift must be placed in exactly one bag
- The capacity of each bag is set at 50 pounds.
- each bag must contain at least three gifts

### Algorithms used
Two types of algorithms have been used to solve the problem: a greedy approach and a genetic algorithm.
Three different versions of the greedy algorithm have been implemented, which fill the bags according to a given rule as long as possible, three rules are followed:
- insert the lighter gifts first
- insert the heaviest gifts first
- the gifts are inserted in the order in which they appear in the initial dataset
Each of the three approaches is content to find the best solution locally.

The genetic algorithm simulates the growth and evolution of a population of individuals. The most promising individuals are selected and paired with each other to produce a new generation. Mutations of individuals are also introduced randomly.
The population represents the set of sacched to be filled, the individuals with greater weight are the most promising and
will survive the selection process.
