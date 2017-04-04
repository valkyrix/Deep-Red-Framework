import logging

from sklearn.preprocessing import normalize

from clusterer_parts.analysis import get_common_features_from_cluster, get_common_feature_stats
from clusterer_parts.clustering import cluster_with_dbscan, cluster_with_kmeans, precompute_distances, \
    cluster_with_agglomerative, cluster_interactive, get_centroids, cluster_single_kmeans, get_k
from clusterer_parts.display import print_cluster_details, generate_dot_graph_for_gephi, create_plot, \
    create_plot_centroids, create_plot_only_centroids, twin, remove_large_clusters
from clusterer_parts.optimizing import sort_items_by_multiple_keys
from clusterer_parts.reduction import pca
from clusterer_parts.validation import validate_clusters, get_average_distance_per_cluster
from clusterer_parts.vectorize import vectorize
import numpy as np
from tabulate import tabulate
firstpass = True

def cluster(
        vector_names,
        vectors,
        reduced_vectors,
        normalized_vectors,
        vectorizer,
        strategy="automatic",
        cluster_method="kmeans",
        n_clusters=2,
        epsilon=0.5,
        min_samples=5,
        metric="euclidean",
):
    """
    Clustering options:

    Manual:
     The user supplies all required information to do the clustering. This includes the clustering algorithm and
     hyper parameters,
     if no cluster count is provided the gap_statistic method will be used to calculate the optimal cluster count

    Assisted:
     The user assists the algorithm by suggesting that some samples should or should not be clustered together

    Automatic:
     The multiple clustering strategies and parameters are used in an attempt to get the best clusters
     
     finds the least amount of clusters with atleast one shared feature
    """

    global centroidskmeans, centroidagglo, centroiddbs, no_clusters, Nno_clusters, Ncentroidskmeans



    if strategy == "manual":
        no_clusters = ""
        if cluster_method == "kmeans":
            #centroidskmeans = get_centroids(reduced_vectors, n_clusters=n_clusters)
            #logging.debug("centroids for kmeans: {0}".format(centroidskmeans))
            return cluster_with_kmeans(reduced_vectors, n_clusters=n_clusters)

        elif cluster_method == "dbscan":
            return cluster_with_dbscan(normalized_vectors, epsilon=epsilon, min_samples=min_samples, metric=metric)

        elif cluster_method == "agglomerative":
            return cluster_with_agglomerative(normalized_vectors, n_clusters=n_clusters, metric=metric)

        else:
            # Unknown clustering method
            raise NotImplementedError()

    elif strategy == "assisted":
        """
        To display a information about a vector to a user, you can use the following:
        display_vector_index_details(vector_index, vectors, vector_names, vectorizer)
        """
        # todo Try with normalized vectors
        return cluster_interactive(reduced_vectors, vectorizer, vectors, vector_names)
    elif strategy == "automatic":
        # todo fix
        results = []
        smallest_cluster_count = vectors.shape[0]
        # centroids works for only kmeans atm
        for cluster_method in [
            "kmeans"  # ,
            # "agglomerative",
            # "dbscan",
        ]:
            if cluster_method == "kmeans":
                #this method is called X-means clustering
                logging.debug("Starting prospective KMeans clusterings")
                move_to_next_method = False
                # start at 2 clusters and end at smallest_cluster_count
                for n_clusters in xrange(2, smallest_cluster_count):
                    logging.debug("Trying {0}".format("kmeans(n_clusters={0})".format(n_clusters)))
                    labels = cluster_with_kmeans(reduced_vectors, n_clusters=n_clusters)
                    overall_score, per_cluster_score = validate_clusters(vectors, labels)
                    mean_distance = get_average_distance_per_cluster(vectors, labels)[0]

                    tsp, msp, msn = get_common_feature_stats(vectors, labels, vectorizer)

                    # If any cluster has 0 shared features, we just ignore the result
                    if msp <= tsp:
                        logging.debug("Not all clusters are informative (a cluster has 0 shared features) ")
                        continue
                    if len(set(labels)) > smallest_cluster_count:
                        move_to_next_method = True
                        # logging.debug("len(set(labels)): {0} > smallest_cluster_count: {1}".format(len(set(labels)), smallest_cluster_count))
                        break
                    if len(set(labels)) < smallest_cluster_count:
                        smallest_cluster_count = len(set(labels))

                    # logging.debug(repr((
                    #         overall_score,
                    #         min(per_cluster_score.values()),
                    #         mean_distance,
                    #         labels,
                    #         len(set(labels)),
                    #         tsp,
                    #         msp,
                    #         msn,
                    #         "kmeans(n_clusters={0})".format(n_clusters)
                    #     )))
                    results.append(
                        (
                            overall_score,
                            min(per_cluster_score.values()),
                            mean_distance,
                            labels,
                            len(set(labels)),
                            tsp,
                            msp,
                            msn,
                            "kmeans(n_clusters={0})".format(n_clusters)
                        )
                    )
                if move_to_next_method:
                    continue

            if cluster_method == "agglomerative":
                logging.debug("Starting prospective Agglomerative clusterings")
                move_to_next_method = False
                for n_clusters in xrange(2, smallest_cluster_count):
                    logging.debug("Trying {0}".format("agglomerative(n_clusters={0})".format(n_clusters)))
                    labels = cluster_with_agglomerative(reduced_vectors, n_clusters=n_clusters, metric=metric)
                    overall_score, per_cluster_score = validate_clusters(vectors, labels)
                    mean_distance = get_average_distance_per_cluster(vectors, labels)[0]

                    tsp, msp, msn = get_common_feature_stats(vectors, labels, vectorizer)

                    # If any cluster has 0 shared features, we just ignore the result
                    if msp <= tsp:
                        logging.debug("Not all clusters are informative (a cluster has 0 shared features) ")
                        continue
                    if len(set(labels)) > smallest_cluster_count:
                        move_to_next_method = True
                        break
                    if len(set(labels)) < smallest_cluster_count:
                        smallest_cluster_count = len(set(labels))

                    logging.debug(repr((
                        overall_score,
                        min(per_cluster_score.values()),
                        mean_distance,
                        labels,
                        len(set(labels)),
                        tsp,
                        msp,
                        msn,
                        "agglomerative(n_clusters={0})".format(n_clusters)
                    )))
                    results.append(
                        (
                            overall_score,
                            min(per_cluster_score.values()),
                            mean_distance,
                            labels,
                            len(set(labels)),
                            tsp,
                            msp,
                            msn,
                            "agglomerative(n_clusters={0})".format(n_clusters)
                        )
                    )
                if move_to_next_method:
                    continue

            if cluster_method == "dbscan":
                logging.debug("Starting prospective DBSCAN clusterings")
                distance_matrix = precompute_distances(vectors, metric=metric)
                min_distance = sorted(set(list(distance_matrix.flatten())))[1]
                max_distance = sorted(set(list(distance_matrix.flatten())))[-1]
                num_steps = 25.0
                step_size = float(max_distance - min_distance) / float(num_steps)
                epsilon = min_distance
                while True:
                    logging.debug("Trying {0}".format("dbscan(epsilon={0})".format(epsilon)))
                    labels = cluster_with_dbscan(reduced_vectors, epsilon=epsilon, min_samples=1,
                                                 distances=distance_matrix)
                    if len(set(labels)) == 1 and list(set(labels))[0] == 0:
                        break
                    overall_score, per_cluster_score = validate_clusters(vectors, labels)
                    mean_distance = get_average_distance_per_cluster(vectors, labels)[0]

                    tsp, msp, msn = get_common_feature_stats(vectors, labels, vectorizer)

                    # If any cluster has 0 shared features, we just ignore the result
                    if msp <= tsp:
                        logging.debug("Not all clusters are informative (a cluster has 0 shared features) ")
                        epsilon += step_size
                        continue

                    logging.debug(repr((
                        overall_score,
                        min(per_cluster_score.values()),
                        mean_distance,
                        labels,
                        len(set(labels)),
                        tsp,
                        msp,
                        msn,
                        "dbscan(epsilon={0})".format(epsilon)
                    )))
                    results.append(
                        (
                            overall_score,
                            min(per_cluster_score.values()),
                            mean_distance,
                            labels,
                            len(set(labels)),
                            tsp,
                            msp,
                            msn,
                            "dbscan(epsilon={0})".format(epsilon)
                        )
                    )
                    epsilon += step_size

        # Pick best result
        """
        We want to maximize the silhouette score while minimizing the number of labels
        """
        sorted_results = sort_items_by_multiple_keys(
            results,
            {
                # 0: True,  # AVG Silhouette
                # 1: True,  # Min Silhouette
                # 2: False,  # Average distance
                4: False,  # Number of clusters
                # 6: True,   # Min common features per cluster
            },
            {
                # 0: 1,
                # 1: 1,
                # 2: 1,
                4: 1,
                # 6: 1
            }
        )
        # logging.debug(sorted_results)
        best_result = results[sorted_results[0][0]]
        # logging.debug(best_result)

        best_method = best_result[-1]
        best_silhouette = best_result[0]
        best_labels = best_result[3]
        global firstpass
        if firstpass:
            no_clusters = best_result[-1]
            firstpass = False
        else:
            Nno_clusters = best_result[-1]

        # no_clusters = best_result[-1]

        logging.info("Best clustering method: {0} (adjusted silhouette == {1})".format(best_method, best_silhouette))
        return best_labels

    else:
        # Unknown strategy
        raise NotImplementedError()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=u'Cluster NMap/Nessus Output')

    parser.add_argument('path', metavar='path', type=str, nargs='+', default=None,
                        help="Paths to files or directories to scan")

    parser.add_argument('-s', '--strategy', default="automatic", choices=["manual", "automatic", "assisted"])
    parser.add_argument('-c', '--method', default="kmeans", choices=["kmeans", "dbscan", "agglomerative"])
    parser.add_argument('--metric', default="euclidean", choices=["euclidean", "cosine", "jaccard"])
    parser.add_argument('-N', '--nessus', default="false", required=False, action='store_true',
                        help='use .nessus file input')

    parser.add_argument('-n', '--n_clusters', type=int, default=2, help='Number of kmeans clusters to aim for')
    parser.add_argument('-e', '--epsilon', type=float, default=0.5, help='DBSCAN Epsilon')
    parser.add_argument('-m', '--min_samples', type=int, default=5, help='DBSCAN Minimum Samples')
    parser.add_argument('-cent', '--centroids', default=False, required=False, action='store_true',
                        help='plot only centroids graph, requires the use of "-p"')
    parser.add_argument('-t', '--twin', default=False, required=False, action='store_true',
                        help='use both input formats to calculate vulnerable single clusters, use with -tp and -N')
    parser.add_argument('-tp', '--twinpath', metavar='twinpath', type=str, required=False,
                        help='path to nmap xml if using twin clustering')

    parser.add_argument('-p', '--plot', default=False, required=False, action='store_true',
                        help='Plot clusters on 2D plane')

    parser.add_argument("-v", "--verbosity", action="count", help="increase output verbosity")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(process)s %(module)s %(funcName)s %(levelname)-8s :%(message)s',
                        datefmt='%m-%d %H:%M')

    if args.verbosity == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbosity > 1:
        logging.getLogger().setLevel(logging.DEBUG)

    if (args.twin == False):

        # Vectorize our input
        logging.info("Vectorizing Stage")
        vector_names, vectors, vectorizer = vectorize(args.path, args.nessus)
        logging.debug("Loaded {0} vectors with {1} features".format(len(vector_names), vectors.shape[1]))
        logging.info("Vectorizing complete")

        # normalise vectors first before passing them through PCA. PCA uses 2 dimensions
        logging.info("Normalising the vectors")
        normalized_vectors = normalize(vectors)
        logging.info("Reducing vectors to two dimensions with PCA")
        reduced_vectors = pca(normalized_vectors)
        logging.debug(
            "reduced to {0} vectors with {1} dimensions".format((reduced_vectors.shape[0]), reduced_vectors.shape[1]))

        # Cluster the vectors
        logging.info("Clustering")
        labels = cluster(vector_names, vectors, reduced_vectors, normalized_vectors, vectorizer, args.strategy,
                         args.method, args.n_clusters, args.epsilon, args.min_samples, args.metric)
        logging.info("Clustering Complete")
        # Test cluster validity
        overall_score, per_cluster_score = validate_clusters(vectors, labels)

        # Analysis relevant to the person reading results
        universal_positive_features, universal_negative_features, shared_features = get_common_features_from_cluster(
            vectors, labels, vectorizer)

        # logging.debug("Shared features: {0}".format(shared_features))

        # Reduce results and relevant information to per cluster data
        cluster_details = {}
        for cluster_id in per_cluster_score.keys():
            cluster_details[cluster_id] = {
                "silhouette": per_cluster_score[cluster_id],
                "shared_positive_features": shared_features[cluster_id]['positive'],
                # "shared_negative_features": shared_features[cluster_id]['negative'],
                "ips": [vector_names[x] for x in xrange(len(vector_names)) if labels[x] == cluster_id]
            }
        print "Note: shared features does not retain keys from XML and therefore wont always be human readable."
        print_cluster_details(cluster_details, shared_features)

        if args.plot:
            # only kmeans centroids for now
            if no_clusters.startswith("kmeans") :
                logging.debug("Getting centroids using reduced vectors:")
                # global centroidskmeans
                # take just cluster number from result string
                n_clusters = no_clusters.split("=", 1)[1]
                n_clusters = int(n_clusters.rsplit(')', 1)[0])
                logging.debug("nclusters: " + str(n_clusters))

                centroidskmeans = get_centroids(reduced_vectors, n_clusters)
                logging.debug("attempting to plot the following centroids:\n " + str(centroidskmeans))

                # covariance
                x = centroidskmeans[:, 0]
                y = centroidskmeans[:, 1]
                X = np.vstack((x, y))
                cov = np.cov(X)
                logging.info("Centroids Covariance Matrix:\n {0}".format(cov))

                # print similarity distance between centroids
                matrix = precompute_distances(centroidskmeans, metric=args.metric)
                matrixTable = tabulate(matrix)
                logging.info(
                    "distance matrix between centroids using metric: {0} :\n{1}".format(args.metric, matrixTable))

                if args.centroids:
                    create_plot_only_centroids(reduced_vectors, labels, vector_names, centroidskmeans, n_clusters)
                else:
                    create_plot_centroids(reduced_vectors, labels, vector_names, centroidskmeans, n_clusters,
                                          cluster_details)

            # manually selected kmeans though arguments
            elif args.method == "kmeans" and args.strategy != "automatic":
                if get_k()>0:
                    centroidskmeans = get_centroids(reduced_vectors, get_k())
                else:
                    centroidskmeans = get_centroids(reduced_vectors, args.n_clusters)

                logging.debug("attempting to plot the following centroids: \n" + str(centroidskmeans))

                # covariance
                x = centroidskmeans[:, 0]
                y = centroidskmeans[:, 1]
                X = np.vstack((x, y))
                cov = np.cov(X)
                logging.info("Centroids Covariance Matrix:\n {0}".format(cov))

                # print similarity distance between centroids
                matrix = precompute_distances(centroidskmeans, metric=args.metric)
                matrixTable = tabulate(matrix)
                logging.info(
                    "distance matrix between centroids using metric: {0} :\n{1}".format(args.metric, matrixTable))

                if args.centroids:
                    if get_k() > 0:
                        create_plot_only_centroids(reduced_vectors, labels, vector_names, centroidskmeans, get_k())
                    else:
                        create_plot_only_centroids(reduced_vectors, labels, vector_names, centroidskmeans, args.n_clusters)
                else:
                    if get_k() > 0:
                        create_plot_centroids(reduced_vectors, labels, vector_names, centroidskmeans, get_k(),
                                              cluster_details)
                    else:
                        create_plot_centroids(reduced_vectors, labels, vector_names, centroidskmeans, args.n_clusters,
                                              cluster_details)
            else:
                logging.debug("plotting standard graph")
                create_plot(reduced_vectors, labels, vector_names)
        # Write DOT diagram out to cluster.dot, designed for input into Gephi (https://gephi.org/)
        with open("cluster.dot", "w") as f:
            f.write(
                generate_dot_graph_for_gephi(precompute_distances(vectors, metric=args.metric), vector_names, labels))

    elif args.twin == True and args.strategy == "automatic":

        logging.debug("twin flag enabled")
        logging.debug("tp: {0} , path: {1}".format(args.twinpath, args.path))

        # Vectorize our input for nessus
        logging.info("Vectorizing Stage for Nessus")
        Nvector_names, Nvectors, Nvectorizer = vectorize(args.path, args.nessus)
        logging.debug("Loaded {0} vectors with {1} features".format(len(Nvector_names), Nvectors.shape[1]))
        logging.info("Vectorizing complete\n")

        # Vectorize our input for nmap
        logging.info("Vectorizing Stage for nmap")
        twinpath = list()
        twinpath.append(args.twinpath)
        vector_names, vectors, vectorizer = vectorize(twinpath, False)
        logging.debug("Loaded {0} vectors with {1} features".format(len(vector_names), vectors.shape[1]))
        logging.info("Vectorizing complete\n")

        # normalise vectors first before passing them through PCA. PCA uses 2 dimensions
        # nessus
        logging.info("Normalising the nessus vectors")
        Nnormalized_vectors = normalize(Nvectors)
        logging.info("Reducing vectors to two dimensions with PCA")
        Nreduced_vectors = pca(Nnormalized_vectors)
        logging.debug(
            "reduced to {0} vectors with {1} dimensions".format((Nreduced_vectors.shape[0]), Nreduced_vectors.shape[1]))
        logging.info("Normalising complete\n")

        # normalise vectors first before passing them through PCA. PCA uses 2 dimensions
        # nmap
        logging.info("Normalising the nmap vectors")
        normalized_vectors = normalize(vectors)
        logging.info("Reducing vectors to two dimensions with PCA")
        reduced_vectors = pca(normalized_vectors)
        logging.debug(
            "reduced to {0} vectors with {1} dimensions".format((reduced_vectors.shape[0]), reduced_vectors.shape[1]))
        logging.info("Normalising complete\n")

        # Cluster the vectors
        logging.info("Clustering Nessus")
        Nlabels = cluster(Nvector_names, Nvectors, Nreduced_vectors, Nnormalized_vectors, Nvectorizer, args.strategy,
                          args.method, args.n_clusters, args.epsilon, args.min_samples, args.metric)
        logging.info("Clustering Complete\n\n")
        # Test cluster validity
        Noverall_score, Nper_cluster_score = validate_clusters(Nvectors, Nlabels)

        # Cluster the vectors
        logging.info("Clustering Nmap")
        labels = cluster(vector_names, vectors, reduced_vectors, normalized_vectors, vectorizer, args.strategy,
                         args.method, args.n_clusters, args.epsilon, args.min_samples, args.metric)
        logging.info("Clustering Complete\n\n")
        # Test cluster validity
        overall_score, per_cluster_score = validate_clusters(vectors, labels)

        # Analysis relevant to the person reading results
        # nessus
        Nuniversal_positive_features, Nuniversal_negative_features, Nshared_features = get_common_features_from_cluster(
            Nvectors, Nlabels, Nvectorizer)

        # Analysis relevant to the person reading results
        # nmap
        universal_positive_features, universal_negative_features, shared_features = get_common_features_from_cluster(
            vectors, labels, vectorizer)

        # Reduce results and relevant information to per cluster data
        # nessus
        Ncluster_details = {}
        for cluster_id in Nper_cluster_score.keys():
            Ncluster_details[cluster_id] = {
                "silhouette": Nper_cluster_score[cluster_id],
                "shared_positive_features": Nshared_features[cluster_id]['positive'],
                "ips": [Nvector_names[x] for x in xrange(len(Nvector_names)) if Nlabels[x] == cluster_id]
            }
        print "Note: shared features does not retain keys from XML and therefore wont always be human readable."
        print "Printing Nessus cluster details\n"
        print_cluster_details(Ncluster_details, Nshared_features)

        print "\n\n"

        # Reduce results and relevant information to per cluster data
        cluster_details = {}
        for cluster_id in per_cluster_score.keys():
            cluster_details[cluster_id] = {
                "silhouette": per_cluster_score[cluster_id],
                "shared_positive_features": shared_features[cluster_id]['positive'],
                # "shared_negative_features": shared_features[cluster_id]['negative'],
                "ips": [vector_names[x] for x in xrange(len(vector_names)) if labels[x] == cluster_id]
            }
        print "Printing Nmap cluster details\n"
        print_cluster_details(cluster_details, shared_features)

        if args.plot:
                # Nmap
                logging.debug("Getting centroids using reduced vectors for Nmap:")
                # take just cluster number from result string
                n_clusters = Nno_clusters.split("=", 1)[1]
                n_clusters = int(n_clusters.rsplit(')', 1)[0])
                logging.debug("nclusters: " + str(n_clusters))

                centroidskmeans = get_centroids(reduced_vectors, n_clusters)
                logging.debug("attempting to plot the following centroids:\n " + str(centroidskmeans) + "\n\n")

                # Nessus
                logging.debug("Getting centroids using reduced vectors for Nessus:")
                # take just cluster number from result string
                logging.debug("nclusters: " + str(Nno_clusters))

                Nn_clusters = no_clusters.split("=", 1)[1]
                Nn_clusters = int(Nn_clusters.rsplit(')', 1)[0])
                logging.debug("nclusters: " + str(Nn_clusters))

                Ncentroidskmeans = get_centroids(Nreduced_vectors, Nn_clusters)
                logging.debug("attempting to plot the following centroids:\n " + str(Ncentroidskmeans) + "\n\n")

                # covariance for Nmap
                x = centroidskmeans[:, 0]
                y = centroidskmeans[:, 1]
                X = np.vstack((x, y))
                cov = np.cov(X)
                logging.info("Nmap Centroids Covariance Matrix:\n {0}".format(cov))

                # covariance for Nessus
                Nx = Ncentroidskmeans[:, 0]
                Ny = Ncentroidskmeans[:, 1]
                NX = np.vstack((Nx, Ny))
                Ncov = np.cov(NX)
                logging.info("Nessus Centroids Covariance Matrix:\n {0}".format(Ncov))

                # print similarity distance between centroids
                # Nessus
                matrix = precompute_distances(centroidskmeans, metric=args.metric)
                matrixTable = tabulate(matrix)
                logging.info(
                    "distance matrix between centroids using metric for Nmap: {0} :\n{1}".format(args.metric, matrixTable))

                # print similarity distance between centroids
                # Nmap
                Nmatrix = precompute_distances(Ncentroidskmeans, metric=args.metric)
                NmatrixTable = tabulate(Nmatrix)
                logging.info(
                    "distance matrix between centroids using metric for Nessus: {0} :\n{1}".format(args.metric, NmatrixTable))

                small_ips = remove_large_clusters()

                logging.info("IP's from clusters with less than 3 IP's:\n {0}".format((small_ips)))

                # nesmap = np.zeros((len(small_ips),4))
                #
                # for index in range(len(small_ips)):
                #     for index2 in range(len(reduced_vectors[:,0])):
                #         if small_ips[index] == vector_names[index2]:
                #             #nesmap[index,0] = small_ips[index]
                #             nesmap[index,0] = reduced_vectors[index2, 0]
                #             nesmap[index,1] = reduced_vectors[index2, 1]
                #     for index3 in range(len(Nreduced_vectors[:,0])):
                #         if small_ips[index] == Nvector_names[index3]:
                #             nesmap[index,2] = Nreduced_vectors[index3, 0]
                #             nesmap[index,3] = Nreduced_vectors[index3, 1]


                #creates large array with 2nd dimension as large enough to hold both feature vectors
                nesmap = np.zeros((len(small_ips), (Nvectors.shape[1]+vectors.shape[1])))


                #for each single ip
                for index in range(len(small_ips)):
                    # for each ip in vectors
                    features = 0
                    for index2 in range(vectors.shape[0]):
                        #if ip is equal to vector ip
                        #logging.debug("if {0} = {1} ".format(small_ips[index], vector_names[index2]))
                        if small_ips[index] == vector_names[index2]:
                            #logging.debug("ip is equal to vector ip")
                            #for every one of this vectors features
                            for index3 in range(vectors.shape[1]):
                                #assign its features to single ip vector
                                nesmap[index,features] = vectors[index2, index3]
                                features +=1
                            break

                    for index2 in range(Nvectors.shape[0]):
                        #logging.debug("if {0} = {1} ".format(small_ips[index], Nvector_names[index2]))
                        if small_ips[index] == Nvector_names[index2]:
                            #logging.debug("ip is equal to vector ip")
                            for index3 in range(Nvectors.shape[1]):
                                #append nessus features onto nmap features
                                nesmap[index,features] = Nvectors[index2, index3]
                                features += 1
                            break

                logging.debug("Loaded {0} vectors with {1} features".format(nesmap.shape[0], nesmap.shape[1]))
                small_normalized_vectors = normalize(nesmap)
                logging.info("Normalizing input and reducing vectors to two dimensions with PCA")
                final = pca(small_normalized_vectors)

                logging.info("Resulting single IP vectors:\n {0}".format(final))

                Smatrix = precompute_distances(final, metric=args.metric)
                SmatrixTable = tabulate(Smatrix)
                logging.info(
                    "distance matrix between centroids of small combined clusters: {0} :\n{1}".format(args.metric, SmatrixTable))
                clusterz = cluster_single_kmeans(final, 2)

                twin(reduced_vectors, labels, vector_names, centroidskmeans, n_clusters, cluster_details, Nreduced_vectors, Nlabels, Nvector_names, Ncentroidskmeans, Nn_clusters, Ncluster_details, small_ips, final, clusterz)

    else: print "not yet implemented #todo"