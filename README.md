#AutomatedSAS - V0.1.1

Author: Graham Roberts
Email: graham.roberts@uconn.edu

Associated paper (preprint): Roberts G, Nieh M-P, Ma A, Yang Q. Automated Structure Analysis of Small Angle Scattering Data via Machine Learning. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-ggnch

Summary: This repository contains a set of tools and tests for conducting automated SAS analysis via a custom hierarchical ML classifier and a ML regression model.

## Key files:

 1.  run_model.py - parses a set of arguments to construct a hierarchical classification model, and calculates its performance on test data  
 2.  baselines.py - evaluates performance using any of the baseline models; either a k-fold cross validation \* or on test data  
 3.  hierarchical.py - set of functions for defining the objects for classifiers  
 4.  sas_krr_reg.py - set of functions needed for the regression model  
 5.  loaders.py - set of utility functions for loading and formatting data  

## run\_model.py  
This script is the bread and butter of both the hierarchical classifier and the regression components as described in the associated paper. 
This loads the data, trains both the classification portion and the regression portion and then evaluates both.
The included script "example_run.sh" will execute the script and provides an example of the call structure.
>python3 src/run_model.py  --datadir data --configdir configs --resultsdir results --evaluate_file ./data/experimental_spectra.csv  

All arguments are keyword arguments, and can be passed in any order, but must include flags.
###arguments
- targets: the list of space separated target morphologies, i.e., 
>--targets cylinder disk sphere cs_cylinder cs_disk cs_sphere

- datadir: A directory containing all the data. There should be a file called "TRAIN_[target].csv" and "TEST_[target].csv" for each target.  
- configdir: The directory containing the configuration files.  
- resultsdir: A directory to save results to.  
- hierarchy_file: A file contaiing the structure of the hierarchical model, should be in the configdir directory.  
- reg_file: A file containing the hyperparameters and targets for the regression models, should be in the configdir directory.  
- extrapolation: A flag for whether to limit the test data to aspect ratios and shell ratios outside the range of the training data.
- evaluate_file: An optional path to a file containing curves to evaluate, this is where to point to new data of interest. Curves must have the same q values.

## baselines.py  
This script contains allows one to run all the baseline comparisons included in the paper.
The script "baselines.sh" shows examples of running each of the included baselines.
It loads the data, trains the off-the-shelf classifier, and evaluates the result.
There are a variety of arguments; many are only applicable to a particular classifier.  
The arguments are as follows:  
###arguments
- targets: The set of morphologies to include, same as above.
- classifier: Which baseline model to use; choose from svc, knn, random-forest.
- datadir: The directory containing the source data.
- k_fold: A flag for whether or not to compare k_fold performance.
- test: A flag for whether or not to evaluate performance on test data.
- extrapolation: A flag for whether to test on all test data or only test data with aspect ratio or shell ratio outside the range of training data.

####parameters for svc
    - c: the c regularization parameter
    - degree: the degree of polynomial if using polynomial kernel
    - gamma: a kernel coefficient, normalized by 1/number of features
    - kernel: which kind of kernel to use, such as 'poly' for polynomial or 'rbf' for radial basis function.
    - coeff0: the coefficient for the intercept of the polynomial

####parameters for knn
    - k: the number of neighbors to use for classification
    - weight: in ['uniform', 'distance'] whether or not to weight the votes from neighbors depending on distance

####parameters for random-forest
    - n_est: the number of estmators
    - rfcriterion: the criterion on which to split
    - max_depth: the maximum depth of the tree
    - min_samples: the minimum number of samples per split

### k_fold
As described in the accompanying paper we opted to perform k_fold with an inverted number of training and validation data.
The data is split into k folds.
For each fold the model is trained on that fold and evaluated on all other folds.
This leads to lower performance on validation, but selects a model that when trained on all data performs well on test data.
This essentially is a form of implicit regularization, looking for models that on small data can still perform moderately well in generalization.
