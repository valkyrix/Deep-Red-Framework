import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from optimal_k_k_means import gap_statistic, elbow_method, find_centers, find_centers_only, optimalK
from Interactive_Clustering import *


def precompute_distances(vectors, metric="euclidean"):
    return pairwise_distances(vectors, n_jobs=-1, metric=metric)


def cluster_interactive(vectors, vectorizer, raw_vectors, vector_names):
    Icluster = interactive_Clustering()
    return Icluster.fit_predict(vectors, vectorizer, raw_vectors, vector_names)

def get_centroids(vectors, n_clusters):
    if n_clusters == 2:
        # Meaning that user didn't set the cluster number, thus we have to find the optimal number of clusters
        # We can choose any of the following 2 methods:
        # n_clusters = elbow_method(vectors, 20);
        k, gapdf = optimalK(vectors, nrefs=10, maxClusters=vectors.shape[0])
        logging.debug("gap statistics recommends number of clusters: {0}\n".format(k))
        if (k > n_clusters):
            n_clusters = k
        else:
            n_clusters = 2

    return find_centers_only(vectors, n_clusters)

def cluster_single_kmeans(vectors, n_clusters):
    if n_clusters == 2:
        # Meaning that user didn't set the cluster number, thus we have to find the optimal number of clusters
        # We can choose any of the following 2 methods:
        #k = elbow_method(vectors, vectors.shape[0]);
        #n_clusters = gap_statistic(vectors, 20)  # 20 is the max amount of allowed clusters
        k, gapdf = optimalK(vectors, nrefs=10, maxClusters=vectors.shape[0])
        logging.debug("gap statistics recommends number of clusters: {0}\n".format(k))
        if (k > n_clusters):
            n_clusters = k
        else:
            n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    return kmeans


def cluster_with_kmeans(vectors, n_clusters):
    if n_clusters == 2:
        # Meaning that user didn't set the cluster number, thus we have to find the optimal number of clusters
        # We can choose any of the following 2 methods:
        #n_clusters = elbow_method(vectors, 20);
        #n_clusters = gap_statistic(vectors, 20)  # 20 is the max amount of allowed clusters
        k, gapdf = optimalK(vectors, nrefs=10, maxClusters=vectors.shape[0])
        logging.debug("gap statistics recommends number of clusters: {0}\n".format(k))
        if (k > n_clusters):
            n_clusters = k
        else:
            n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
    return kmeans.fit_predict(vectors)


def cluster_with_dbscan(vectors, epsilon=0.5, min_samples=5, distances=None, metric="euclidean"):
    # precomputing our distances will be faster as we can use multiple cores
    if distances is None:
        distances = pairwise_distances(vectors, n_jobs=-1, metric=metric)

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric="precomputed")
    return dbscan.fit_predict(distances)


def cluster_with_agglomerative(vectors, n_clusters=2, metric="euclidean"):
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete", affinity=metric)
    return agg.fit_predict(vectors)
