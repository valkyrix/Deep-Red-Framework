#### clusterer

Manual:
     choose between kmeans, agglomerative and dbscan. all values used in the clustering process can be inputed via arguments.

Assisted:
     The user assists the algorithm by suggesting that some samples should or should not be clustered together. A yes/no type scenario.

Automatic:
     The multiple clustering strategies and parameters are used in an attempt to get the best clusters
	 

manual e.g. using kmeans with 3 clusters and plot results into a graph on a nmap xml file: 

'''
cluster.py -s manual -c kmeans -n 3 -p nmap.xml
'''

automatic e.g. on a nessus .nessus XML file: 

'''
cluster.py -s automatic -N results.nessus
'''

arguments usage:
'''
usage: cluster.py [-h] [-s {manual,automatic,assisted}]
                  [-c {kmeans,dbscan,agglomerative}]
                  [--metric {euclidean,cosine,jaccard}] [-N] [-n N_CLUSTERS]
                  [-e EPSILON] [-m MIN_SAMPLES] [-p] [-v]
                  path [path ...]
'''

