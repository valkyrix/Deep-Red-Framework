# Deep-Red-Framework

## Implementing Deep Learning techniques to construct a Red Teaming solution

### Tasks

Items in __bold__ are currently being worked on.
Items ~~ruled~~ are completed.

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
* Make the software portable (currently everything is being setup in a non-portable environment). This involves nessus, metasploit, postgresql, theano/anaconda and linux environments to be taken into account.
* ~~The possibility of using virtual environments such as docker to run all tools required. Using a windows host creates a small issue as mentioned below.~~
* ~~Building dockerfile with ML tools and metasploit ,nessus. this should dramatically reduce prototype time even if its restricted to cpu for now.~~ Dockerfile can be found on docker resource git res- https://github.com/valkyrix/res-   
* Project documentation (Thesis/Journal).

__current problem: figuring out what to make the RNN calculate, what features and labels it should have. Deciding on what vulnerability to focus on for a POC__
Whether or not the label should be severity/chance of exploitation or just exploitable (yes/no). How to decide these labels based on limited datasets.

The CPU version (Dockerfile.cpu) will run on all the above operating systems. However, the GPU version (Dockerfile.gpu) will only run on Linux OS. This is because Docker runs inside a virtual machine on Windows and OS X. Virtual machines don't have direct access to the GPU on the host. Unless PCI passthrough is implemented for these hosts, GPU support isn't available on non-Linux OSes at the moment.

#### Please check back soon for more details.

