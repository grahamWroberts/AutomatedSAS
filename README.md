AutomatedSAS - V0.1
Author: Graham Roberts

Associated paper (preprint): Roberts G, Nieh M-P, Ma A, Yang Q. Automated Structure Analysis of Small Angle Scattering Data via Machine Learning. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-ggnch

Summary: This repository contains a set of tools and tests for conducting automated SAS analysis via a custom hierarchical ML classifier and a ML regression model.

Key files:
 1.  run_model.py - parses a set of arguments to construct a hierarchical classification model, and calculates its performance on test data
 2.  baselines.py - evaluates performance using any of the baseline models; either a k-fold cross validation \* or on test data
 3.  hierarchical.py - set of functions for defining the objects for classifiers
 4.  sas_krr_reg.py - set of functions needed for the regression model
 5.  loaders.py - set of utility functions for loading and formatting data


