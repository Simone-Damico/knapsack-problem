# coding: utf-8

"""
preprocessing.py

Module for data preprocessing.

From the dataset with only gifts to a dataset with four columns:

--------------------------------------------
| GiftId | type_toy | id_toy | weights_toy |
--------------------------------------------
|    .   |     .    |    .   |       .     |
|    .   |     .    |    .   |       .     |
|    .   |     .    |    .   |       .     |
--------------------------------------------

Imported modules:
- pandas
- numpy

Functions:
- create_weight(pandas.Dataframe) -> None
- preprocessing(string) -> pandas.Dataframe
- create_instance(int) -> None
"""
import pandas as pd
import numpy as np


def create_weight(df):
    """
    Creation of weights for gifts. Each gift is assigned a weight based on the distributions given:
    - horse = max(0, np.random.normal(5,2,1))
    - ball = max(0, 1 + np.random.normal(1,0.3,1))
    - bike = max(0, np.random.normal(20,10,1))
    - train = max(0, np.random.normal(10,5,1))
    - coal = 47 * np.random.beta(0.5,0.5,1)
    - book = np.random.chisquare(2,1)
    - doll = np.random.gamma(5,1,1)
    - block = np.random.triangular(5,10,20,1)
    - gloves = 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe of gifts

    Returns
    -------
    list
        list of weights assigned to gifts

    Raises
    ------
    ValueError
        if df is not an instance of pandas.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be an instance of pandas.DataFrame")
    res = []
    type_toy = pd.unique(df['type_toy']) # Type of gifts
    for t in type_toy:
        keep_going = True
        n_elem = len(df.loc[df['type_toy'] == t]) # Counting gifts of the same type
        # Assignment of weights
        if t == 'horse':
            while keep_going:
                weights = np.maximum(0, np.random.normal(5, 2, n_elem))
                if all(weights <= 50):
                    res.extend(weights)
                    keep_going = False

        elif t == 'ball':
            while keep_going:
                weights = np.maximum(0, 1 + np.random.normal(1, 0.3, n_elem))
                if all(weights <= 50):
                    res.extend(weights)
                    keep_going = False

        elif t == 'bike':
            while keep_going:
                weights = np.maximum(0, np.random.normal(20, 10, n_elem))
                if all(weights <= 50):
                    res.extend(weights)
                    keep_going = False

        elif t == 'train':
            while keep_going:
                weights = np.maximum(0, np.random.normal(10, 5, n_elem))
                if all(weights <= 50):
                    res.extend(weights)
                    keep_going = False

        elif t == 'coal':
            while keep_going:
                weights = 47 * np.random.beta(0.5, 0.5, n_elem)
                if all(weights <= 50):
                    res.extend(weights)
                    keep_going = False

        elif t == 'book':
            while keep_going:
                weights = np.random.chisquare(2, n_elem)
                if all(weights <= 50):
                    res.extend(weights)
                    keep_going = False

        elif t == 'doll':
            while keep_going:
                weights = np.random.gamma(5, 1, n_elem)
                if all(weights <= 50):
                    res.extend(weights)
                    keep_going = False

        elif t == 'blocks':
            while keep_going:
                weights = np.random.triangular(5, 10, 20, n_elem)
                if all(weights <= 50):
                    res.extend(weights)
                    keep_going = False

        elif t == 'gloves':
            while keep_going:
                weights = 3.0 + np.random.rand(1, n_elem)[0] if np.random.rand(1) < 0.3 else np.random.rand(1, n_elem)[0]
                if all(weights <= 50):
                    res.extend(weights)
                    keep_going = False

    return res

def preprocessing(path):
    """
    Preparation of the dataset:
     - extrapolation type of gift
     - added columns of ID, type and weight of gifts

    Parameters
    ----------
    path : string
        Path from which to load the dataset

    Returns
    -------
    pandas.DataFrame
        dataframe after the preprocessing

    Raises
    ------
    FileNotFoundError
        Invalid path or file not found
    """

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("Invalid path or file not found")

    # Extrapolation of the type of toy from the name of the toy itself and definition of the id
    type_toy = []
    id_toy = []
    for ind, elem in enumerate(df["GiftId"]):
        id_toy.append(ind)
        type_toy.append(elem.split("_")[0])

    # Add the toy_type column with its weight
    df["type_toy"] = type_toy
    df["id_toy"] = id_toy
    df["weights_toy"] = create_weight(df)
    return df

def create_instances(path, name_df=None):
    """
    Creation of dataset with the correct distributions to be used to test the algorithms.
    The dataset is saved if specified.

    Parameters
    ----------
    path : str
        Path of gifts dataset

    save_df : bool
        Flag for save dataframe

    name_df : str
        Name of dataset
    """

    df = preprocessing(path)
    if name_df is not None:
        df.to_csv("dataset/{}.csv".format(name_df), index=False)
    print("Dataset created")
    return df

