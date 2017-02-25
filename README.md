# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* exploration of quora-question-pair data-set
* word2vec - given text-data, trains word-vectors and visualizes them via connected components.
* perform deep question pair classification 

### How do I get set up? ###

* Clone repo
* Install Anaconda (to setup virtual env and load dependencies)
* cd into repo
* conda env create -f environment.yml
* source activate deep-nlp

# Install more packages
* (cd nlp && git clone --recursive https://github.com/dmlc/xgboost.git  && cd xgboost && sh build.sh && cd python-package && python3 setup.py install --user )
* (cd nlp && git clone https://github.com/fmfn/BayesianOptimization)
