AutomatedSAS - V0.1
Author Graham Roberts
This is a very early version of this codebase. In the 1.0 release, a much better user interface as well as beter class and function definiteions will be provided to help users implement this into their own work. For now this one allows recreation of the results in the paper. 
A set of tools and tests for conducting an automated SAS analysis via a custom hierarchical ML classifier and a ML regression
There are a few notable files present
 1.  run_model.py - which is a script that parses some arguments to construct a hierarchical model at calculates its performance on test data. 
 2.  baselines.py - A script which evaluates performance using any of the baseline models, either a k-fold cross validation \* or on test data.
 3.  hierarchical.py - which is a set of functions for defining the objects for classifiers.
 4.  sas_krr_reg.py - The set of functions needed for the regression portion.
 5.  loaders.py - a set of utility functions for loading data and formatting it correctly.


