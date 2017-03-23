# Deep-Red-Framework

## Implementing Deep Learning techniques to construct a Red Teaming solution

### Tasks

Items in __bold__ are currently being worked on.
Items ~~ruled~~ are completed.

#### Clustering Model

Using the university's hacklab as a dataset with nmap-clustering the following results were acheived. 
More information can be found within the hacklab analyses folder.

clustering model on automatic mode, using hacklab nessus scan resulting in kmeans with 7 clusters:

<img src="/hacklab analyses/hacklab_nessus_kmeans_7c.png" width="500">

also on automatic but clustering the nmap -A output this time resulting in using dbscan with an epsilon of 4.14:

<img src="/hacklab analyses/hacklab_nmap_auto_dbscan_ep4.png" width="500">

Click the following for clustering information:

* [Nessus, auto mode](/hacklab%20analyses/hacklab_nessus_kmeans_7c.txt) 
* [NMAP, auto mode](/hacklab%20analyses/hacklab_nmap_dbscan_ep4.txt) 

#### todo

* ~~Network capturer~~
* ~~Feature extration and clustering~~
* ~~Consider labels~~
* ~~Create a nessus xml file parser~~ code is messy but it works and is easily editable to incorporate any aspect of nessus file as features
* ~~Nessus cluster~~ only usable nessus file i have at the moment is my home network which is small and not very clusterable however it seems to work relatively well.

Using a nessus file from my home network scan.
![perf_test](https://s13.postimg.org/xtrm6cehz/nessus_home_agglo_3c.png)

192.168.0.4 and 192.168.0.7 are both recent windows machines whilst 192.168.0.1 is the home router and 192.168.0.8 is an android device.

* __Compare clusters from nmap to a nessus equivelant__
* Integration with exploitation model



#### Please check back soon for more details.

