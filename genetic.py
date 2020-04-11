# coding: utf-8

"""
genetic.py

Ogni elemento del vettore può assumere valore 1 se i-esimo regalo è inserito nella sacca 0 altrimenti.
Per selezionare gli elementi migliori si calcola il peso degli oggetti inseriti, più è vicino alla capacità
della sacca più l'individuo è promettente.
Ad ogni generazione, l'algoritmo valuta la popolazione e seleziona gli individui più promettenti,
li fà accoppiare e si produce una nuova generazione.
Ogni individuo può, in aggiunta, mutare con una certa probabilità.

Module for the implementation of the generic algorithm.
The algorithm simulates the growth of a population of individuals.
Each individual is a vector as long as the number of gifts to be placed in the bags.
Each element of the vector can take value 1 if the i-th gift is inserted in the bag or 0 otherwise.
To select the best elements, the weight of the inserted objects is calculated, the closer it is to the capacity
of the bag more the individual is promising.
At each generation, the algorithm evaluates the population and selects the most promising individuals,
it makes them mate and a new generation is produced.
In addition, each individual can mutate with a certain probability.

Imported modules
- numpy
- pandas
- random
- time
- os

Class:
- Genetic

"""
import numpy as np
import pandas as pd
import random
import time
import os

class Genetic:
    """
    Genetic algorithm implementation class.

    Attributes
    ----------
    dataframe : pandas.DataFrame
        Dataframe of gifts
    dim_pop : int
        Population size
    num_gen : int
        Number of generations to consider
    path_out : string
        Path where to save the simulation result

    Functions:
    - __init__(pandas.DataFrame, int, int, string) -> Genetic
    - create_init_pop() -> list
    - mutate(numpy.array, int) -> None
    - fitness(list) -> float
    - evolve_pop(list, float, float, int) -> list
    - get_toys(numpy.array, bool) -> list, numpy.array
    - simulation() -> None
    """

    def __init__(self, dataframe, dim_pop, num_gen, path_out=None, seed=None):
        """
        Constructor of the Genetics class.

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe of gifts
        dim_pop: int
            Population size
        num_gen: int
            Number of generations to consider
        path_out: string
            Path where to save the simulation result
        seed : int
            Seed for random choice
        capacity: int
            Maximum capacity of the bag

        Raises
        ------
        ValueError
             if df is not an instance of pandas.DataFrame
         ValueError
             if dim_pop is not an integer
         ValueError
             if num_gem is not an integer
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be an instance of pandas.DataFrame")
        if not isinstance(dim_pop, int):
            raise ValueError("dim_pop must be an integer")
        if not isinstance(num_gen, int):
            raise ValueError("num_gen must be an integer")

        self.df = dataframe
        self.dim_pop = dim_pop
        self.num_gen = num_gen
        self.path_out = path_out
        self.seed = seed
        self.capacity = 50

    def create_init_pop(self):
        """
        Creation of the initial population randomly with at least three selected gifts.

        Returns
        -------
        list
            Initial population
        """
        pop = []
        for _ in range(0, self.dim_pop):
            perm = np.zeros(len(self.df))
            if self.seed is not None:
                random.seed(self.seed)
            ind_ones = np.random.randint(1, len(self.df), 3)
            for i in ind_ones:
                perm[i] = 1
            pop.append(perm)
        return pop

    def mutate(self, perm, n_change):
        """
        Mutation of an individual.
        Select n elements of the permutation to change, deciding which to remove and which to put in the bag.

        Parameters
        ----------
        perm : numpy.array
            Permutation to be changed
        n_change: int
            number of elements to be mutate

        Raises
        ------
        ValueError
            if n_change is not an integer
        """
        if not isinstance(n_change, int):
            raise ValueError("n_change must be an integer")

        # if n_change is greater than the length of the permutation then change all the elements
        if n_change > len(perm):
            n_change = len(perm)

        # Extracting the indices to be changed
        if self.seed is not None:
            random.seed(self.seed)
        ind_to_change = np.random.choice(len(perm), n_change, replace=False)
        for i in ind_to_change:
            if perm[i] == 1:
                perm[i] = 0
            else:
                perm[i] = 1

    def fitness(self, perm):
        """
        Calculate the fitness value of individuals as the sum of the weights of the objects considered.
        Higher the fitness value is, more the individual is promised in the population.
        If the weight of the objects exceeds the capacity from the bag then the permutation is not valid
        and its fitness value will be -1.
        It cannot be 0 because there can be gifts with a weight equal to 0.

        Parameters
        ----------
        perm : numpy.array
            Permutation to be mutate

        Returns
        -------
        float
            permutation weight
        """
        weight = 0
        for ind, toy in enumerate(perm):                # For each gift in the permutation
            if toy == 1:                                # if its value is 1
                weight += self.df.iloc[ind].weights_toy # increase the weight of the permutation with weight of the gift

        # The permutation is not valid because its weight is greater than the maximum capacity of the bag
        if weight > self.capacity:
            return -1
        # The permutation is valid
        else:
            return weight

    def evolve_pop(self, pop, survival_threshold=25.0, mutation_chance=0.2, mutation_n=5):
        """
        Population evolution:
        1) Evaluation of the population
        2) Selection of the most promising individuals
        3) Any mutations between individuals
        4) Coupling of individuals and generation of children
        5) Any mutations of the children

        Parameters
        ----------
        pop : list
            Population for evolution
        survival_threshold : float
            Threshold for the survival of the population, is the minimum weight that each individual must have to survive
        mutation_chance : float
            Probability of mutation of the individual
        mutation_n : int
            Elements to be changed in the individual

        Returns
        -------
        list
            new population

        Raises
        ------
        ValueError
            The survival threshold of the population cannot be greater than the capacity of the bag
        ValueError
            The probability of mutation must be between 0 and 1
        """

        # Controllo degli argomenti in input
        if survival_threshold > self.capacity:
            raise ValueError("The survival threshold of the population cannot be greater than the capacity of the bag")
        if not 0 <= mutation_chance <= 1:
            raise ValueError("The possibility of mutation must be between 0 and 1")

        parents = []
        # Selection of combinations with fitness greater than the threshold
        for perm in pop:
            if self.fitness(perm) >= survival_threshold:
                parents.append(perm)

        # If there are no promising individuals, a new population is created
        if len(parents) == 0:
            parents = self.create_init_pop()

        # Parental mutation
        for p in parents:
            if self.seed is not None:
                random.seed(self.seed)
            if mutation_chance > random.random():
                self.mutate(p, mutation_n)

        # Combination of permutation and generation of children
        children = []

        # Number of children to be created to keep the population size constant
        n_new_elem = len(pop) - len(parents)
        while len(children) < n_new_elem:
            # Selection of parents
            if self.seed is not None:
                random.seed(self.seed)
            parent_1 = pop[random.randint(0, len(parents) - 1)]
            if self.seed is not None:
                random.seed(self.seed)
            parent_2 = pop[random.randint(0, len(parents) - 1)]

            # Creation of the child
            if self.seed is not None:
                random.seed(self.seed)
            ind = random.randint(0, len(parent_1) - 1)

            # A part from the first parent and a part from the second
            child = np.concatenate([parent_1[:ind], parent_2[ind:]])

            # Possible mutation of the child
            if self.seed is not None:
                random.seed(self.seed)
            if mutation_chance > random.random():
                self.mutate(child, mutation_n)

            # If the child contains at least three elements it is accepted
            if np.count_nonzero(child) < 3:
                perm = np.zeros(len(self.df))
                if self.seed is not None:
                    random.seed(self.seed)
                ind_ones = np.random.randint(1, len(self.df), 3)
                for i in ind_ones:
                    perm[i] = 1

            children.append(child)

        # Adding children to the population
        parents.extend(children)
        return parents

    def get_toys(self, perm, indexes_too=False):
        """
        Given a permutation, it returns the gifts chosen for insertion and possibly their indexes

        Parameters
        ----------
        perm : numpy.array
            Permutation to be examined
        indexes_too : bool
            Flag to also obtain the gift indexes

        Returns
        -------
        list
            list of gifts in the bag
        numpy.array
            list of gift indexes
        """

        ind_elem = np.where(perm == 1)
        res = self.df.iloc[ind_elem].GiftId.values
        if indexes_too:
            return res, ind_elem
        return res

    def simulation(self):
        """
        Simulation management and invocation of other functions.
        Writing the simulation results dataset.
        """
        # Initialization of the result dataset
        res = pd.DataFrame(columns=['id', 'elements', 'weight', 'n_elem', 'time'])
        bag = -1

        # Loop over the toys to insert
        while len(self.df) > 1:
            bag += 1
            start_time = time.time()
            pop = self.create_init_pop()
            for gen in range(0, self.num_gen):
                pop = self.evolve_pop(pop)
            best_element = sorted(pop, key=lambda x: self.fitness(x), reverse=True)[0]
            fit = self.fitness(best_element)

            # Identification of the individual mile of the population at the end of evolution
            eleme_x_bag, ind_elem_drop = self.get_toys(best_element, indexes_too=True)
            self.df = self.df.drop(self.df.index[ind_elem_drop])
            end_time = time.time()

            # Inclusion of the individual with its data in the data frame
            res = res.append({'id': bag,
                              'elements': eleme_x_bag,
                              'weight': fit,
                              'n_elem': len(eleme_x_bag),
                              'time': end_time - start_time}, ignore_index=True)

        # If there are still toys to be inserted, I add them to the first bag
        if not self.df.empty:
            res = res.append({'id': bag + 1,
                              'elements': self.df.GiftId.values,
                              'weight': self.df.weights_toy.values[0],
                              'n_elem': len(self.df),
                              'time': 0.0},
                             ignore_index=True)

        # Write the result dataset
        if self.path_out is not None:
            if not os.path.isdir("result"):
                os.mkdir("result")
            res.to_csv("result/{}.csv".format(self.path_out), index=False)

