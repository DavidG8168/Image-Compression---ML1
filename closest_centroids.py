import numpy as np


# Find the closest centroids.
def closest_centroids(X, c):
    K = np.size(c, 0)
    idx = np.zeros((np.size(X, 0), 1))
    arr = np.empty((np.size(X, 0), 1))
    for i in range(0, K):
        y = c[i]
        temp = np.ones((np.size(X, 0), 1)) * y
        b = np.power(np.subtract(X, temp), 2)
        a = np.sum(b, axis=1)
        a = np.asarray(a)
        a.resize((np.size(X, 0), 1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr, 0, axis=1)
    idx = np.argmin(arr, axis=1)
    return idx
