import numpy as np
import numpy.matlib


# Calculate the centroids.
def compute_centroids(X, idx, K):
    n = np.size(X, 1)
    centroids = np.zeros((K, n))
    for i in range(0, K):
        ci = idx == i
        ci = ci.astype(int)
        total_number = sum(ci)
        ci.resize((np.size(X, 0), 1))
        total_matrix = np.matlib.repmat(ci, 1, n)
        ci = np.transpose(ci)
        total = np.multiply(X, total_matrix)
        centroids[i] = (1/total_number) * np.sum(total, axis=0)
    return centroids
