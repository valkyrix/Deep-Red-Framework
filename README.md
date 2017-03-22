# Deep-Red-Framework

## Implementing Deep Learning techniques to construct a Red Teaming solution

### Tasks

Items in __bold__ are currently being worked on.
Items ~~ruled~~ are completed.

#### Clustering Model (Priority) 

Using the university's hacklab as a dataset with nmap-clustering the following results were acheived. 
More information can be found within the hacklab analyses folder.

clustering model on automatic using hacklab nessus scan resulting in kmeans with 7 clusters:
![hacklab nessus kmeans 7c]('/hacklab analyses/hacklab_nessus_kmeans_7c.png'?raw=true "hacklab nessus kmeans 7c")

[click here]('/hacklab analyses/hacklab_nessus_kmeans_7c.txt') for the full information regarding this clustering.


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

