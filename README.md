# Deep-Red-Framework

## Implementing Deep Learning techniques to construct a Red Teaming solution

### Tasks

Items in __bold__ are currently being worked on.
Items ~~ruled~~ are completed.

#### Clustering Model

Using the university's hacklab nmap and nessus scans as a dataset the following results were acheived. 
More information can be found within the hacklab analyses folder.

clustering model on automatic mode with centroids using nmap xml output (left) and nessus scan output (right)
the vectors have been normalised before PCA(2d).

<img src="/hacklab analyses/nmap_nessus_respect.png">

the covariance matrices for each are as follows:

```
centroids covariance matrix for nmap output:
 [[ 0.23001336  0.02479788]
 [ 0.02479788  0.14788991]]
 
centroids covariance matrix for nessus output:
 [[ 0.13216597  0.00459265]
 [ 0.00459265  0.08221746]]
```

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
* __Compare the two clusters and generate a new cluster based off of them__
* automate the entire process not just clustering (scanning and retreival)
* possible integration with exploitation model

Click the following for clustering information without normalisation before PCA (old):

* [Nessus, auto mode](/hacklab%20analyses/hacklab_nessus_kmeans_7c.txt) 
* [NMAP, auto mode](/hacklab%20analyses/hacklab_nmap_dbscan_ep4.txt) 

individual readme.md's for each model:
* [clusterer](/Clusterer/README.md) 
* [exploitation model](/exploit%20system/README.md) 