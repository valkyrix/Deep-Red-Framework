### Changelog 6:39am 4th May 17
* changed background colour of GUI to white, its more pleasing to the eyes.
* implemented fix for bug when user runs manual mode without specifying cluster amount on kmeans. It crashes 1/5 times but works otherwise. I assume this is due to the problem of time stepping and can probably be fixed by sleeping threads at a certain point, or by the gap statistic returning very large cluster numbers.
* fixed manual mode in general, works as intended now.
* improved general efficiency
* ~~BUG: when using automatic mode and gap statistic is selected by the algorithm, the centroids will not be displayed correctly on the basic clusterings although the combined clustering will function as intended.~~ FIXED

## Differential IP clustering for Vulnerability Detection using applied Machine Learning

#### Clustering Model

Using the university's hacklab nmap and nessus scans as a dataset the following results were acheived. 
More information can be found within the hacklab analyses folder.

twin mode: requires both datasets as input. This will take clusters with IP count < 3 and combine the vector features of both datasets to create a new clustering of 'vulnerable' IP's. In other words - Small clusterings (IP count less than 3) reclustered with combined datasets.

<img src="/hacklab analyses/twin.png">

command:
```syntax
cluster.py -s automatic -vv -p -t -N -tp "../hacklab analyses/hacklab_new.xml" "../hacklab analyses/hacklab_new.nessus"
```
full raw output which includes all relevant information can be found here:  [twin-auto-output.txt](/hacklab%20analyses/twin-auto-output.txt) 

clustering model on automatic mode with centroids using nmap xml output (left) and nessus scan output (right)
the vectors have been normalised before PCA(2d).

<img src="/hacklab analyses/nmap_nessus_respect.jpg">

the covariance matrices for each are as follows:

```
centroids covariance matrix for nmap output:
 [[ 0.23001336  0.02479788]
 [ 0.02479788  0.14788991]]
 
centroids covariance matrix for nessus output:
 [[ 0.13216597  0.00459265]
 [ 0.00459265  0.08221746]]
```
### Usage

```syntax
usage: cluster.py [-h] [-s {manual,automatic,assisted}]
                  [-c {kmeans,dbscan,agglomerative}]
                  [--metric {euclidean,cosine,jaccard}] [-N] [-n N_CLUSTERS]
                  [-e EPSILON] [-m MIN_SAMPLES] [-cent] [-t] [-tp twinpath]
                  [-p] [-v]
                  path [path ...]
Cluster NMap/Nessus Output
positional arguments:
  path                  Paths to files or directories to scan
optional arguments:
  -h, --help            show this help message and exit
  -s {manual,automatic,assisted}, --strategy {manual,automatic,assisted}
  -c {kmeans,dbscan,agglomerative}, --method {kmeans,dbscan,agglomerative}
  --metric {euclidean,cosine,jaccard}
  -N, --nessus          use .nessus file input
  -n N_CLUSTERS, --n_clusters N_CLUSTERS
                        Number of kmeans clusters to aim for
  -e EPSILON, --epsilon EPSILON
                        DBSCAN Epsilon
  -m MIN_SAMPLES, --min_samples MIN_SAMPLES
                        DBSCAN Minimum Samples
  -cent, --centroids    plot only centroids graph, requires the use of "-p"
  -t, --twin            use both input formats to calculate vulnerable single
                        clusters, use with -tp and -N
  -tp twinpath, --twinpath twinpath
                        path to nmap xml if using twin clustering
  -p, --plot            Plot clusters on 2D plane
  -v, --verbosity       increase output verbosity
```

### Tasks

Items in __bold__ are currently being worked on.
Items ~~ruled~~ are completed.
#### todo

* ~~Network capturer~~
* ~~Feature extration and clustering~~
* ~~Consider labels~~
* ~~Create a nessus xml file parser~~ code is messy but it works and is easily editable to incorporate any aspect of nessus file as features
* ~~Nessus cluster~~ only usable nessus file i have at the moment is my home network which is small and not very clusterable however it seems to work relatively well.

Using a nessus file from my home network scan.
<img src="https://s13.postimg.org/xtrm6cehz/nessus_home_agglo_3c.png" width="500">
192.168.0.4 and 192.168.0.7 are both recent windows machines whilst 192.168.0.1 is the home router and 192.168.0.8 is an android device.

* ~~calculate covariance matrix of centroids~~ only works for kmeans due to agglo and dbscan python implementations not returning centroids
* ~~Compare the two clusters and generate a new cluster based off of them~~ takes all ips in clusters less than 3 ip's in size and combines each ips data from the 2 data sets and clusters based off of gap_statistic k-means

#### Stretch Goals (after thesis completion)
* automate the entire process not just clustering (scanning and retreival)
* possible integration with exploitation model

Click the following for clustering information without normalisation before PCA (old):

* [Nessus, auto mode](/hacklab%20analyses/hacklab_nessus_kmeans_7c.txt) 
* [NMAP, auto mode](/hacklab%20analyses/hacklab_nmap_dbscan_ep4.txt) 

individual readme.md's for each model:
* [clusterer](/Clusterer/README.md) 
* [exploitation model](/exploit%20system/README.md) 
