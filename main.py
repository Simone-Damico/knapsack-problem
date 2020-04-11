# coding: utf-8

"""
test.py

Module for the execution of the algorithms.
For each instance of the problem, the greedy algorithm in its three variants and the genetic algorithm are executed

Imported modules
- pandas
- genetic
- greedy

"""
import time
import preprocessing
from genetic import Genetic
from greedy import Greddy


n_instances = 2

for i in range(0, n_instances):
    t1 = time.time()
    df = preprocessing.create_instances("gifts.csv", "instance_{}".format(i+1))
    print("Dataset {} in processing".format(i+1))

    print("Version in ascending order")
    test_greedy_asc = Greddy(df, method="ascending", path_out="test_greedy_asc_{}".format(i+1))
    test_greedy_asc.greedy_algorithm()

    print("Version in descending order")
    test_greedy_disc = Greddy(df, method="descending", path_out="test_greedy_disc_{}".format(i+1))
    test_greedy_disc.greedy_algorithm()

    print("Version with no sorting")
    test_greedy_no_ord = Greddy(df, path_out="test_greedy_no_ord_{}".format(i+1))
    test_greedy_no_ord.greedy_algorithm()

    print("Genetic algorithm starts execution")
    test_GA_3 = Genetic(df, dim_pop=50, num_gen=50, path_out="test_GA_{}".format(i+1), seed=123)
    test_GA_3.simulation()

    print("Total time for instance #{}: {}".format(i+1, time.time()- t1))

print("End of execution")





