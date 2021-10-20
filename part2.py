#!/bin/python
import numpy as np
from tqdm import trange
from movielens import *
from typing import Tuple
from collections import deque
from sklearn.metrics import mean_squared_error


# Store data in arrays
user = []
item = []
rating = []
rating_test = []

# Load the movie lens dataset into arrays
d = Dataset()
d.load_users("data/u.user", user)
d.load_items("data/u.item", item)
d.load_ratings("data/u.base", rating)
d.load_ratings("data/u.test", rating_test)

n_users = len(user)
n_items = len(item)

# The utility matrix stores the rating for each user-item pair in the matrix form.
# Note that the movielens data is indexed starting from 1 (instead of 0).
utility = np.zeros((n_users, n_items))
for r in rating:
    utility[r.user_id-1][r.item_id-1] = r.rating

# Finds the average rating for each user and stores it in the user's object
for i in range(n_users):
    rated = np.nonzero(utility[i])
    n = len(rated[0])
    if n != 0:
        user[i].avg_r = np.mean(utility[i][rated])
    else:
        user[i].avg_r = 0.

print(utility)


def pcs(x, y):
    """Finds the Pearson Correlation Similarity Measure between two users."""

    x_ratings = utility[x.id - 1]
    y_ratings = utility[y.id - 1]

    common_ratings = np.intersect1d(np.flatnonzero(x_ratings), np.flatnonzero(y_ratings))

    if len(common_ratings) == 0:
        return 0
    else:
        x_diff = x_ratings[common_ratings] - x.avg_r
        y_diff = y_ratings[common_ratings] - y.avg_r
        pcs_dividend = (x_diff * y_diff).sum()
        pcs_divisor = np.sqrt((x_diff**2).sum()) * np.sqrt((y_diff**2).sum())

        return pcs_dividend / pcs_divisor


def guess(user_id, i_id, top_n):
    """
    Guesses the ratings that user with id, user_id, might give to item with id, i_id. We will consider the top_n similar
    users to do this.
    """
    _u = user[user_id - 1]
    top_users: deque[Tuple[float, int]] = deque(maxlen=top_n)

    for u in user:
        if u.id != user_id and utility[u.id - 1][i_id - 1] > 0:
            if len(top_users) == top_n:
                top_users = deque(sorted(list(top_users) + [(pcs(_u, u), u.id)]), maxlen=top_n)
            else:
                top_users.append((pcs(_u, u), u.id))

    #print(f"top users for {user_id - 1}: {top_users}")
    top_utility = utility[[t[1] - 1 for t in top_users]]
    norm_top_rating = (top_utility - np.expand_dims(top_utility.mean(axis=1), 1))[:, i_id - 1].mean()

    return _u.avg_r + norm_top_rating


# Display the utility matrix as given in Part 1 of your project description
np.set_printoptions(precision=3)

# Finds the average rating for each user and stores it in the user's object
print("computing user avg ratings")
for i in trange(n_users):
    rated = np.nonzero(utility[i])
    n = len(rated[0])
    if n != 0:
        user[i].avg_r = np.mean(utility[i][rated])
    else:
        user[i].avg_r = 0.

n = 150 # Assume top_n users

# Finds all the missing values of the utility matrix
print("\nfilling utility matrix")
utility_copy = np.copy(utility)
for i in trange(n_users):
    for j in range(n_items):
        if utility_copy[i][j] == 0:
            utility_copy[i][j] = guess(i+1, j+1, n)

# Test without clustering
print("testing performance")
guesses, real_ratings = tuple(zip(*[(guess(r.user_id, r.item_id, n), r.rating) for r in rating_test]))
# Finds the mean squared error of the ratings with respect to the test set
print("Mean Squared Error is " + str(mean_squared_error(guesses, real_ratings)))

## THINGS THAT YOU WILL NEED TO DO:
# Perform clustering on users and items
# Predict the ratings of the user-item pairs in rating_test
# Find mean-squared error
