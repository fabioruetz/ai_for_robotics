import numpy as np
import random
from scipy import misc
from scipy import ndimage
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image
import sys


def main():

    # Settings for k_means
    n_clusters = 3

    # Load precomputed (PCA) features from file and normalize data
    features = np.loadtxt("features_k_means.txt")
    features = preprocessing.scale(features)
    n_samples, n_features = np.shape(features)

    for i in range(1):
        clusters, mu = k_means(features,n_clusters,True)
        plot(clusters,mu)
    # Save results.
    print("Found {} clusters witch cord {}".format(mu.shape[0],mu))
    np.savetxt('results_k_means.txt', mu)



def k_means(features, n_clusters, k_pp =False):
    # Input:
    #       features: (mxn) np.array m: #data points, n: #features
    #       n_clusters: # of clusters
    #       k_pp: Initialisation for k-means++ if true
    # Return:
    #       clusters: dic[k]{lxn np.array} l: #points in clsuter k, n: #feat.
    #       mu: np.array (n_clust.x n)  Contains the mu of each cluster k


    #Initialize cluster centres as mu
    mu = initialisation(features,n_clusters,k_pp)
    # Perform Lloyd's algorithm.
    mu, clusters = find_centers(features, mu)

    return clusters, mu


def find_centers(X,mu):

    oldmu = np.zeros(mu.shape[0])

    while not has_converged(mu, oldmu):
        oldmu = mu
        # First step of optimization: Assign all datapoints in X to clusters.
        clusters = clustering(X, mu)
        # Second step of optimization: Optimize location of cluster centers mu.
        mu = reevaluate_mu(oldmu, clusters)
    return (mu, clusters)

def clustering(X, mu):
    # Assign each point to a cluster k by minimal distance (eucl) to mu[k]
    clusters = {k: np.array((1, 1)) for k in range(mu.shape[0])}
    # Distance matrix of all points with respect to each mu
    D = cdist(X, mu, 'euclidean')
    # Find the minmum value for each point and save the cluster label
    idx_min = np.argmin(D, axis=1)
    # Seperate the points into clusters
    for k in range(mu.shape[0]):
        id_k = np.argwhere(idx_min == k)
        cluster_k = X[id_k[:], :]
        cluster_k = np.reshape(cluster_k, (id_k.shape[0], X.shape[1]))
        clusters[k] = cluster_k

    return clusters
 
def reevaluate_mu(mu, clusters):
    # Calculate the new mu for each cluster
    new_mu = np.zeros(mu.shape)
    for i in range(mu.shape[0]):
        new_mu[i,:] = np.mean(clusters[i], axis=0)
    return new_mu
 
def has_converged(mu, oldmu):
    # Check if the old and new mue are the same
    return np.array_equal(mu,oldmu)

def initialisation(features,n_clusters,k_pp):

    idx = np.zeros((n_clusters))
    # Initialize the mu for all clusters and check that each mu is unique
    while not np.unique(idx).size is len(idx):
        if k_pp is True:
            mu,idx = k_means_pp(features,n_clusters)
        else:
            idx = np.random.choice(np.arange(features.shape[0]),n_clusters)
            mu = features[idx,:]
    return mu

def k_means_pp(features,n_clusters):
    # K-mean++ initialisation

    # Variables for mu and id_mu
    mu = np.zeros((n_clusters,features.shape[1]))
    idx_mu = np.zeros(n_clusters,dtype=int)
    # Choose first mu randomly
    idx_mu[0] = np.random.choice(np.arange(features.shape[0]), 1)
    mu[0,:] = features[idx_mu[0], :]
    # Pick the i <= k mu with prob proportional to the distance all i-1 mu
    for i in range(1,n_clusters):
        # Compute distance of every point with every mu
        D = cdist(features, mu[0:i, :], 'euclidean')
        # Id of the minimal distance between mu and a point
        idx_min = np.argmin(D, axis=1)
        # Compute probability of picking this center
        prob = D[np.arange(0, features.shape[0]), idx_min]
        prob = prob / np.sum(prob)
        # Choose i'th mu randomly with a prob proportional on distance
        idx_mu[i] = np.random.choice(np.arange(features.shape[0]), 1,p=prob)
        mu[i, :] = features[idx_mu[i], :]

    return mu, idx_mu

def plot(clusters,mu):
    # Plot all clusters and mue for cord_1 and cord_2
    # Todo: Add loop for to plot all clusters -> collor array
    cluster_1 = clusters[0]
    cluster_2 = clusters[1]
    cluster_3 = clusters[2]
    plt.plot(cluster_1[:, 0], cluster_1[:, 1], 'o', markersize=7,
             color='blue', alpha=0.5, label='Cluster 1')
    plt.plot(cluster_2[:, 0], cluster_2[:, 1], 'o', markersize=7,
             color='black', alpha=0.5, label='Cluster 2')
    plt.plot(cluster_3[:, 0], cluster_3[:, 1], 'o', markersize=7,
             color='green', alpha=0.5, label='Cluster 3')
    plt.plot(mu[:,0], mu[:,1],'*', markersize=20, color='red', alpha=1.0,
             label='Cluster centers')
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.title('K-means clustering')
    plt.show(block=False)

if __name__ == '__main__':
    main()
    sys.exit()