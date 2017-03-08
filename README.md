# Deep-Red-Framework

## Implementing Deep Learning techniques to construct a Red Teaming solution

### Tasks

Items in __bold__ are currently being worked on.
Items ~~ruled~~ are completed.

#### Clustering Model (Priority)

__will probably use nmap clustering project from blackhat (incomplete)__ found at https://github.com/CylanceSPEAR/NMAP-Cluster  - Got working with kmeans and agglomerative but automated is broken.

Using the university's hacklab as a dataset with nmap-clustering the following results were acheived.
Using PCA followed by Agglomerative with 3 clusters:
![plot_test](https://s10.postimg.org/57f0nvoe1/hacklab_agglomerative_3clusters.png)

which relates to the following:

```
Cluster ID: 0
Silhouette Score: 0.380005107148
IPs: 10.0.0.209, 10.0.0.24, 10.0.0.25, 10.0.0.3, 10.0.0.30, 10.0.0.32, 10.0.0.33, 10.0.0.36, 10.0.0.38, 10.0.0.40, 10.0.0.44, 10.0.0.45, 10.0.0.49, 10.0.0.5, 10.0.0.50, 10.0.0.52, 10.0.0.53, 10.0.0.55, 10.0.0.59, 10.0.0.61, 10.0.0.62, 10.0.0.66, 10.0.0.67, 10.0.0.68, 10.0.0.69, 10.0.0.72, 10.0.0.73, 10.0.0.74, 10.0.0.75, 10.0.0.76, 10.0.0.77, 10.0.0.79, 10.0.0.80, 10.0.0.82, 10.0.0.83, 10.0.0.85

Shared Features:


Cluster ID: 1
Silhouette Score: -0.142359054524
IPs: 10.0.0.1, 10.0.0.190, 10.0.0.7

Shared Features:
tcp - 80 - open - http


Cluster ID: 2
Silhouette Score: 1.0
IPs: 10.0.0.171

Shared Features:
tcp - 2000 - open - cisco-sccp
tcp - 3000 - open - http - Thin
tcp - 3000 - open - http - http-server-header -  - thin
tcp - 3000 - open - http - http-title - title -  Solicitors and Estate Agents in Dundee | Edinburgh | Perth | Angus | Fife
```

####the possibility of using the university hacklab as an nmap dataset for clustering?

* Network capturer
* Feature extration and clustering
* Consider labels
* Compare traffic captured features to network scan features (?)
* Integration with exploitation model

#### Exploitation Model (Backlogged)

* ~~Do background research on Machine Learning models and finalise design~~
* Find or create a legal dataset (Shodan api?). Rapid7 had a project called Sonar which spawned scans.io (very interesting).
* ~~Automate the sanitization and collation of data (CVE's etc)~~
* ~~Standardize data from network reports in order to input into RNN.~~ Nessus now exports via csv's.
* __Create overall outlining framework.__
* __Random Forest and RNN integration. Possibly using Keras ontop of Theano.__
* Train the models.
* Test the models.
* ~~Metasploit cli integration. Find a way to script meterpreter into python scripts.~~ note: msgrpc module needs to be running in order to script metasploit (needs to be executed manually on windows systems). Possibility of using rc files on linux instead of python.
* Automate metasploit exploit checker based off of random forest results.
* Technical report creation section
* ~~Make the software portable (currently everything is being setup in a non-portable environment). This involves nessus, metasploit, postgresql, theano/anaconda and linux environments to be taken into account.~~ Nessus reports get taken from docker share folder, which is the only program not being internally run by the docker container.
* ~~The possibility of using virtual environments such as docker to run all tools required. Using a windows host creates a small issue as mentioned below.~~
* ~~Building dockerfile with ML tools and metasploit ,nessus. this should dramatically reduce prototype time even if its restricted to cpu for now.~~ The incomplete Dockerfile can be found on docker resource git res- https://github.com/valkyrix/res-   
* Project documentation (Thesis/Journal).

__current task:__ ~~figuring out what to make the RNN calculate, what features and labels it should have. Deciding on what vulnerability to focus on for a POC
Whether or not the label should be severity/chance of exploitation or just exploitable (yes/no). How to decide these labels based on limited datasets.
Testing out the possibility of using metasploitable 2/3 as a test machine for the framework. Considering its build to test security tools.~~  (Exploitation)

Programming a model which will capture the network traffic during a specific attack and cluster features into graph, compare these features to those of a normal network scan to detect whether some computers have similar enough features to be considered vulnerable. (Clustering)

__Issues and considerations:__

* Cant use Metasploitable 2/3 due to it only working on Vmware/vbox which does not run with Hyper-v is enabled (required for docker on windows) therefore docker and metasploitable 2 cannot be run concurrently on w10. Using secondary system to run Vagrant metasploitable 3 on the network from Arch linux. 

* Downsides to using metasploit are that it has no support for common Web application tests such as SQL injection and input tampering.

* The CPU version (Dockerfile.cpu) will run on all the above operating systems. However, the GPU version (Dockerfile.gpu) will only run on Linux OS. This is because Docker runs inside a virtual machine on Windows and OS X. Virtual machines don't have direct access to the GPU on the host. Unless PCI passthrough is implemented for these hosts, GPU support isn't available on non-Linux OSes at the moment.

The possibility of using multiple plugins for metasploit to aid time constraints, such as this one: https://github.com/darkoperator/Metasploit-Plugins

docker theano performance test:
![perf_test](https://s18.postimg.org/b4ajbnh55/docker_perf_test.jpg)



#### Please check back soon for more details.

