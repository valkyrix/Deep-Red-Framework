import numpy as np
import pandas as pd
from numpy import zeros
from sklearn.cluster import KMeans


def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i] - c) ** 2 / (2 * len(c)) \
                for i in range(K) for c in clusters[i]])


def find_centers_only(X, K):
    kmeans = KMeans(n_clusters=K);
    mu = kmeans.fit(X).cluster_centers_;
    clusters = cluster_points(X, mu);
    #return (mu, clusters)
    return mu

def find_centers(X, K):
    kmeans = KMeans(n_clusters=K);
    mu = kmeans.fit(X).cluster_centers_;
    clusters = cluster_points(X, mu);
    return (mu, clusters)

def gap_statistic(X, k_max):
    '''
    Algorithm described in paper: http://web.stanford.edu/~hastie/Papers/gap.pdf
    :param X: data matrix
    :param k_max: the maximum number of clusters
    :return: the optimal number of clusters
    '''
    oldGap = 0;
    num_cluster = 0;
    for k in xrange(1, k_max + 1):
        mu, clusters = find_centers(X, k)
        Wks = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = zeros(B)
        for i in range(B):
            Xb = []
            for n in range(X.shape[0]):
                Xb.append(np.random.uniform(0, 1, size=X.shape[1]));
            Xb = np.array(Xb);
            mu, clusters = find_centers(Xb, k);
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs = sum(BWkbs) / B
        sk = np.sqrt(sum((BWkbs - Wkbs) ** 2) / B)
        Gap = Wkbs - Wks;
        if Gap - oldGap <= sk:
            num_cluster = k;
            break;
        oldGap = Gap;
    return num_cluster;


def elbow_method(X, k_max):
    '''
    Algorithm: Elbow Method ("http://blog.geomblog.org/2010/03/this-is-part-of-occasional-series-of.html")
    :param X: data matrix
    :param k_max: the maximum number of clusters
    :return: the optimal number of clusters
    '''
    num_cluster = 0;
    threshold = 0.01;
    mu, clusters = find_centers(X, 1)
    WksOld = np.log(Wk(mu, clusters));
    for k in xrange(2, k_max + 1):
        mu, clusters = find_centers(X, k)
        Wks = np.log(Wk(mu, clusters));
        if WksOld - Wks < threshold:  # We take the log of the SSE because we want to compare the relative change
            num_cluster = k;
            break;
        WksOld = Wks;
    return num_cluster;


def optimalK(data, nrefs, maxClusters):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal