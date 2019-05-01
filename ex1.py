import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

from scipy.misc import imread
from init_centroids import init_centroids
from closest_centroids import closest_centroids
from run_kMean import run_kMean


def main():
    # data preparation (loading, normalizing, reshaping)
    path = 'dog.jpeg'  # supplied by instructor.
    A = imread(path)  # load image.
    A = A.astype(float) / 255.  # normalize and reshape.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])

    # variables.
    rows = img_size[0]
    cols = img_size[1]
    channels = img_size[2]
    iterations = 10  # number of iterations.

    # 2 clusters.
    K1 = 2
    initial_centroids = init_centroids(X, K1)
    centroids, idx = run_kMean(X, initial_centroids, iterations)
    idx = closest_centroids(X, centroids)
    X_recovered = centroids[idx]
    X_recovered = np.reshape(X_recovered, (rows, cols, channels))
    scipy.misc.imsave('dog_2.jpg', X_recovered)

    # 4 clusters.
    K2 = 4
    initial_centroids = init_centroids(X, K2)
    centroids, idx = run_kMean(X, initial_centroids,iterations)
    idx = closest_centroids(X, centroids)
    X_recovered = centroids[idx]
    X_recovered = np.reshape(X_recovered, (rows, cols, channels))
    scipy.misc.imsave('dog_4.jpg', X_recovered)

    # 8 clusters.
    K3 = 8
    initial_centroids = init_centroids(X, K3)
    centroids, idx = run_kMean(X, initial_centroids, iterations)
    idx = closest_centroids(X, centroids)
    X_recovered = centroids[idx]
    X_recovered = np.reshape(X_recovered, (rows, cols, channels))
    scipy.misc.imsave('dog_8.jpg', X_recovered)

    # 16 clusters case.
    K4 = 16  # clusters.
    initial_centroids = init_centroids(X, K4)
    centroids, idx = run_kMean(X, initial_centroids, iterations)
    idx = closest_centroids(X, centroids)
    X_recovered = centroids[idx]
    X_recovered = np.reshape(X_recovered, (rows, cols, channels))
    scipy.misc.imsave('dog_16.jpg', X_recovered)


if __name__ == "__main__":
    main()
