import logging

from analysis import reduce_shared_features, get_common_features_from_cluster
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

def display_vector_index_details(vector_index, vectors, vector_names, vectorizer):
    # borrow the feature -> string method from get_common_features_from_cluster
    output = ""
    upf, unf, shared_features = get_common_features_from_cluster(vectors[vector_index, :].reshape(1, -1), np.array([0]), vectorizer)
    ip = vector_names[vector_index]
    features = set()
    for feature in reduce_shared_features(shared_features[0]['positive']):
        features.add(" - ".join([i.encode("unicode-escape") for i in feature]))

    features = sorted(list(features))
    output += "IP: {0}\n".format(ip)
    for f in features:
        output += f + "\n"
    return output + "\n"


def display_shared_vector_indeces_details(vector_indeces, vectors, vector_names, vectorizer):
    # borrow the feature -> string method from get_common_features_from_cluster
    output = ""
    upf, unf, shared_features = get_common_features_from_cluster(vectors[vector_indeces, :], np.array([0] * len(vector_indeces)), vectorizer)

    features = set()
    for feature in reduce_shared_features(shared_features[0]['positive']):
        features.add(" - ".join([i.encode("unicode-escape") for i in feature]))

    features = sorted(list(features))
    if len(features) == 0:
        return "No shared features\n"
    for f in features:
        output += f + "\n"
    return output + "\n"


def print_cluster_details(cluster_details, shared_features):
    amount=0
    for cluster_id in cluster_details.keys():
        amount += 1

    global clusterX
    clusterX = np.zeros((amount, 3), dtype=object)
    #clusterX is a 2d array that includes the first ip, silhouette and clusterid
    for cluster_id in cluster_details.keys():
        print "Cluster ID: {0}".format(cluster_id)
        print "Silhouette Score: {0}".format(cluster_details[cluster_id]["silhouette"])
        print "IPs: {0}".format(", ".join(sorted(cluster_details[cluster_id]["ips"])))
        print ""
        print "Shared Features:"
        for feature in reduce_shared_features(shared_features[cluster_id]['positive']):
            print "".join([i.encode("unicode-escape") for i in feature])

        print ""
        print ""

        clusterX[cluster_id,0] = cluster_id
        clusterX[cluster_id,1] = cluster_details[cluster_id]["silhouette"]
        #calculate the first ip in the cluster
        fip = "{0}".format(",".join(sorted(cluster_details[cluster_id]["ips"])))
        #logging.debug("ID: {0} FIP: {1}".format(cluster_id, fip))
        #fip2 = fip.split(',', 1)[0] #gets just first IP
        ipList = fip.rsplit(',')
        #logging.debug("IP address ipList: {0}".format(ipList))
        #sets 3rd value on y axis as a IP list
        clusterX[cluster_id,2] = ipList

        #print clusterX[cluster_id,2]



def generate_dot_graph_for_gephi(distance_matrix, vector_names, labels, graph_name="nmapcluster"):
    dot_string = "digraph nmapcluster {\n"
    max_distance = distance_matrix.max()
    for index, vn in enumerate(vector_names):
        dot_string += "{0} [label=\"{2} - {1}\" cluster_id=\"{2}\" size=\"64\"];\n".format(index, vn, labels[index])

    for x in xrange(len(vector_names) - 1):
        for y in xrange(x + 1, len(vector_names)):
            dist = ((max_distance - distance_matrix[x, y]) / max_distance)
            if dist <= 0:
                dist = 0.00001
            dot_string += "{0} -> {1} [weight=\"{2}\"]\n".format(x, y, dist)

    dot_string += "}\n"
    return dot_string

def find_nearest_vector(array, value):
        #this method finds the nearest vector to given coords

        #old function, doesnt work so well using numpy
        # idx = np.array([np.linalg.norm(x+y) for (x,y) in array-value]).argmin()
        # return array[idx]
        #spacial works much better for this purpose
        nearest_vector = array[spatial.KDTree(array).query(value)[1]]
        return nearest_vector

def create_plot_centroids(reduced_vectors, labels, vector_names, centroids, n_clusters, cluster_details):

        logging.debug("ploting graph with centroids.")
        plt.figure()
        colors = plt.get_cmap('jet')(np.linspace(0, 1.0, len(set(labels))))
        # colors2 = plt.get_cmap('jet')(np.linspace(0, 1.0, n_clusters))

        # ip's
        for index in range(len(labels)):
            # plt.scatter(reduced_vectors[index, 0], reduced_vectors[index, 1], c=colors[labels[index]], label=labels[index])
            plt.scatter(reduced_vectors[index, 0], reduced_vectors[index, 1], c=colors[labels[index]],zorder=1)
            #annotate ip's
            plt.annotate(vector_names[index], xy=(reduced_vectors[index, 0], reduced_vectors[index, 1]), color="0.8")
        IPList = list()
        #centroids
        for index2 in range(n_clusters):

            clusterID=99
            pt =(centroids[index2, 0], centroids[index2, 1])
            # find reduced vector X and Y of nearest IP
            nearest = find_nearest_vector(reduced_vectors,pt)
            #use nearest reduced vector to find nearest IP
            for index3 in range(len(labels)):
                if nearest[0] == reduced_vectors[index3, 0] and nearest[1] == reduced_vectors[index3, 1]:
                    #found nearest IP to cluster
                    ip = vector_names[index3]
                    #logging.debug("nearest IP to cluster {0} is {1}. nearest vector X: {2}  Y: {3}".format(index2,ip, nearest[0], nearest[1]))
                    break
                continue
            # find cluster_id of the nearest IP
            for index4 in range(n_clusters):
                # if first IP of cluster is inside this clusters ip list

                IPlist = clusterX.item((index4, 2))
                if ip in IPlist:
                    # set clusterID to the one from the nearest IP
                    clusterID = clusterX.item((index4,0))

                    #logging.debug("setting clusterno to: {0} from IP: {1}".format(clusterID, ip))
                    #logging.debug("IPlist from {0} : {1}".format(index4, IPlist))

            # plt.scatter((centroids[index2, 0]), (centroids[index2, 1]), marker='x', s=150, linewidths=2, \
            #             c=colors2[index2], cmap=plt.matplotlib.cm.jet)

            #plot graph points for centroids
            plt.scatter((centroids[index2, 0]), (centroids[index2, 1]), marker='+', s=300, linewidths=1, \
                        c="black", label=index2, zorder=2)


            plt.annotate(clusterID, xy=(centroids[index2, 0], centroids[index2, 1]), color="b")

        #plt.legend(loc='best')
        plt.title("Clustering of IP's with Centroids after Vectorisation.")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


def create_plot(reduced_vectors, labels, vector_names):

        logging.debug("ploting graph without centroids.")
        plt.figure()
        colors = plt.get_cmap('jet')(np.linspace(0, 1.0, len(set(labels))))
        for index in range(len(labels)):
            plt.scatter(reduced_vectors[index, 0], reduced_vectors[index, 1], c=colors[labels[index]], label=labels[index])
            plt.annotate(vector_names[index], xy=(reduced_vectors[index, 0], reduced_vectors[index, 1]))
        #plt.legend(loc='best')
        plt.show()
