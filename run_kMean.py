import numpy as np
import math
from closest_centroids import closest_centroids
from compute_centroids import compute_centroids


# Run the 10 kmeans iterations and print the correct output string.
def run_kMean(X, initial_centroids, iterations):
    m = np.size(X, 0)
    n = np.size(X, 1)
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids
    idx = np.zeros((m, 1))
    print("k={}:".format(K))
    for i in range(0, iterations + 1):
        idx = closest_centroids(X, centroids)
        nameStr = "iter {}: {}".format(i, ", ".join(str(x) for x in (np.floor(centroids * 100)/100).tolist()))
        nameStr = nameStr.replace("0.0,", "0.,")
        nameStr = nameStr.replace("0.0],", "0.],")
        print(nameStr)
        nameStr = 0
        centroids = compute_centroids(X, idx, K)
    return centroids, idx
