# coding: utf-8

"""
greedy.py

Module for the implementation of the greedy algorithm in the three variants:
- descending order: the heavier gifts are placed in the bags first
- increasing order: lighter gifts are placed in the bags first
- no ordering: gifts to order in the dataset are inserted into the bags

Imported modules
- time
- pandas
- os

Class:
- Greddy
"""

import time
import os
import pandas as pd

class Greddy:
    """
    Class of the greedy algorithm.

    Attributes
    ----------
    dataframe : pandas.DataFrame
        Dataframe of gifts
    method : string
        Method of ordering gifts
    path_out : string
        Path where to save the simulation result

    Functions:
    - __init__(pandas.DataFrame, string, string) -> Greedy
    - greedy_algorithm() -> None
    """

    def __init__(self, dataframe, method="no_order", path_out=None):
        """
        Constructor of the Greedy class

        Parameters
        ----------
        toy_list_heavy : pandas.DataFrame
            Dataframe of gifts weighing more than 0
        toy_list_light : pandas.DataFrame
            Dataframe of gifts weighing less than 0
        method : string
            Method of ordering gifts
        capacity : int
            Maximum capacity of the bag
        path_out : string
            Path where to save the simulation result

        Raises
        ------
        ValueError
            if method is not descending, ascending or no_order
        """
        if method == "descending":
            self.toy_list_heavy = dataframe[dataframe.weights_toy > 0].sort_values(by=['weights_toy'], ascending=False)
        elif method == "ascending":
            self.toy_list_heavy = dataframe[dataframe.weights_toy > 0].sort_values(by=['weights_toy'], ascending=True)
        elif method == "no_order":
            self.toy_list_heavy = dataframe[dataframe.weights_toy > 0]
        else:
            raise ValueError("method must be descending, ascending or no_order (default choice)")
        self.toy_list_light = dataframe[dataframe.weights_toy == 0]
        self.method = method
        self.capacity = 50
        self.path_out = path_out


    def greedy_algorithm(self):
        """
        Greedy algorithm
        The gifts are placed in the bags according to their sorting and weight,
        they are removed from the list of gifts still to be inserted.
        Insert the gifts with weight 0 in the bags with less than 3 elements
        """

        # Greedy algorithm for inserting gifts into bags
        res = pd.DataFrame(columns=['id', 'elements', 'weight', 'n_elem', 'time'])
        n_bag = -1
        # Keep inserting until I put all the presents
        while not self.toy_list_heavy.empty:
            start_time = time.time()
            bag = []
            n_bag += 1
            capacity = self.capacity
            keep_going = True
            # Continue to insert into the bag as long as there are gifts with a weight
            # less than the residual weight to be inserted
            while keep_going:
                elem_sort = self.toy_list_heavy[self.toy_list_heavy.weights_toy <= capacity]
                if not elem_sort.empty:
                    elem_to_insert = elem_sort.iloc[0]
                    # It continues as long as there are gifts with a weight lower than the residual weight to be inserted
                    self.toy_list_heavy = self.toy_list_heavy.drop(elem_to_insert.id_toy)
                    bag.append(elem_to_insert.GiftId)
                    capacity -= elem_to_insert.weights_toy
                else:
                    keep_going = False
                    # The gifts with weight 0 are inserted in the bags with less than 3 elements
                    if len(bag) < 3:
                        elem_sort = self.toy_list_light[self.toy_list_light.weights_toy <= capacity]
                        if not elem_sort.empty:
                            elem_to_insert = elem_sort.head(3 - len(bag))
                            self.toy_list_light = self.toy_list_light.drop(elem_to_insert.id_toy)
                            bag.extend(elem_to_insert.GiftId.values)
                            capacity -= elem_to_insert.weights_toy

            end_time = time.time()
            res = res.append({'id': n_bag,
                              'elements': bag,
                              'weight': 50 - capacity,
                              'n_elem': len(bag),
                              'time': end_time - start_time}, ignore_index=True)

        # Any gifts with weight 0 left in the first bag are inserted
        if not self.toy_list_light.empty:
            old_val = res.iloc[0].elements
            new_val = old_val.extend(self.toy_list_light.type_toy.values.tolist())
            res.at[0, 'elements'] = new_val


        if self.path_out is not None:
            if not os.path.isdir("result"):
                os.mkdir("result")
            res.to_csv("result/{}.csv".format(self.path_out), index=False)

