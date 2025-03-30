#AutomatedSAS v0.1.1
#Author: Graham Roberts

#run_model.py a script for training and evaluating the full hierarchical classifier
import numpy as np
import pandas as pd
import sys
from os.path import join as join_path
sys.path.append('..')
import loaders
import argparse
from sklearn.svm import SVC
from sklearn.metrics import classification_report as CR
from sklearn.metrics import accuracy_score as AS
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import hierarchical as hier
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score, regression_mean_width_score
import time

#parse args
#reads in option flags when script is invoked
def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--targets', default = ['cylinder', 'disk', 'sphere', 'cs_cylinder', 'cs_disk', 'cs_sphere'], nargs = '+')
   parser.add_argument('--datadir', default = './data', help = 'the directory where the raw data are stored')
   parser.add_argument('--configdir', default = '../configs', help = 'the directory where the configuration files for the classifier and regressor are stored')
   parser.add_argument('--resultsdir', default = '../results', help = 'the directory where results and logs will be stored')
   parser.add_argument('--hierarchy_file', default="hierarchical_structure.txt")
   parser.add_argument('--reg_file', type = str, default='krr_hyperparameters.txt')
   parser.add_argument('--extrapolation', type=bool,default=False)
   parser.add_argument('--evaluate_file', type=str, default=None, help='a file containing curves to predict, this is where to pass new curves without labels to evaluate')
   parser.add_argument('--quotient', type=bool, default=False)
   parser.add_argument('--uncertainty', type=bool, default = False)
   return(parser.parse_args())
   
#construct regressor
#Arguments: pfile a filepath to a file contining the regressor hyperparaeters for each structural parameter for each target
# gamma norm: a normalization factor, usually just num features
# Returns: A dictionary of dictionaries of dictionaries of instantiated regression objects
#   {morphology 1:{structural parameter 1: {regression object},
#                  structural parameter 2: {regression object},
#                  ...},
#    morphology 2:{structural parameter 1: {regression object},
#                  structural parameter 2: {regression object},
#                  ...},
#    ...}  
def construct_regressor(pfile, gamma_norm):
    rdict = {}
    pdict = parse_pfile(pfile)
    for t in pdict.keys():
        if t not in rdict.keys():
            rdict[t] = {}
        for p in pdict[t].keys():
            if pdict[t][p]['kernel'] == 'polynomial':
                regressor = KRR(alpha = pdict[t][p]['alpha'], gamma = pdict[t][p]['gamma']/gamma_norm, degree = pdict[t][p]['degree'], kernel = pdict[t][p]['kernel'], coef0 = pdict[t][p]['coef0'])
            else:
                regressor = KRR(alpha = pdict[t][p]['alpha'], gamma = pdict[t][p]['gamma']/gamma_norm, kernel = pdict[t][p]['kernel'].lower(), coef0 = pdict[t][p]['coef0'])
            rdict[t][p] = regressor
    return(rdict)

#parse pfile
#Arguments: pfile a filepath to a file containing the hyperparameters for each structural parameter for each target
# Returns: A dictionary of dictionaries of dictionaries of hyperparameters for each regression
#   {morphology 1:{structural parameter 1: {hyperparameters for regression},
#                  structural parameter 2: {hyperparameters for regression},
#                  ...},
#    morphology 2:{structural parameter 1: {hyperparameters for regression},
#                  structural parameter 2: {hyperparameters for regression},
#                  ...},
#    ...}  
def parse_pfile(pfile):
   data_conversions = {'alpha':float,'gamma':float,'kernel':str,'degree':int,'coef0':float}
   pdict = {}
   infile = open(pfile, 'r')
   for line in infile.readlines():
       tokens = line.split()
       ps = {}
       for i in range(len(tokens)):
           t = tokens[i]
           if t.replace('-','').strip() == 'target':
               targ = tokens[i+1].strip()
           elif t.replace('-','').strip() == 'param':
               p = tokens[i+1].strip()
           elif t[:2] == '--':
               pname = tokens[i].replace('-','').strip()
               pf = data_conversions[pname]
               ps[pname] = pf(tokens[i+1])
       if targ not in pdict.keys():
           pdict[targ] = {}
       pdict[targ][p]=ps
   return(pdict)

#train all regression
#Trains each of the regression objects
#Arguments train_curves: a dictionary mapping each target to an array of curves
#          train_params: a dictionary of dictionarries of arrays, mapping each morphology, to each structural parameter to an array of values
#          regressors: a dictionary of dictionaries of regression objects to be trained
#Returns: trained regrrssors dictionary
def train_all_regression(train_curves, train_params, regressors):
    for t in train_params.keys():
        for p in regressors[t].keys():
            regressors[t][p] = regressors[t][p].fit(train_curves[t], train_params[t][p])
    return(regressors)

#This is a helper function definining the specific structure of this hierarchical classifier
#in 1.0 this functionality will be offloaded onto a json structure
# This function defines the structure of the hierachy for the classifier
# each decision is a dictionary that maps the number of a morphology to its label in the hierarchiacal version
#   i.e., decision one is the decision between [0,1,3,4] and [2,5], in this case [cylinder, disk, cs_cylinder, cs_disk] and [sphere, cs_sphere]
# hierarchical map shows that for each output of a decsion, where it should be sent to.
#   anython predicted 0 by the first decision goes to decision 2, anything predicted 1 foes to decision 1.
#   anything predicted as 0 by decision 1 is class '2' and anything predicted 1 is class '5', the ones sent to decision 2 are further decided into  [0, 1] vs [3, 4]
# hierarchical_map, and deciions are the encodng of the hierarchical classifier structure
def hierarchical_definition():
    decision1 = {0:0,1:0,2:1,3:0,4:0,5:1}
    decision2 = {2:0,5:1}
    decision3 = {0:0,1:0,3:1,4:1}
    decision4 = {0:0,1:1}
    decision5 = {3:0,4:1}
    decisions = [decision1, decision2, decision3, decision4, decision5]
    hierarchical_map = [{0:2,1:1},{0:'2',1:'5'},{0:3,1:4},{0:'0',1:'1'},{0:'3',1:'4'}]
    return(hierarchical_map, decisions)

def calibrate_regression_UQ(regressors, X_cal, y_cal):
    uncertainty_regressors = {}
    for t in regressors.keys():
        uncertainty_regressors[t] = {}
        for p in regressors[t].keys():
           mapie = MapieRegressor(regressors[t][p], test_size = 0.3, method = "plus", cv=10)
           mapie.fit(X_cal[t], y_cal[t][p])
           uncertainty_regressors[t][p] = mapie
    return(uncertainty_regressors)

# compare regression is a large function that does many things
# taking the predicted labels from the classifier it passes each test curve to the respective set of regression objects
# it then writes out a set of files to compare the predicted and labeled parameter values for each curve
# since some curves are mispredicted those are separated into different files so one can compare the outputs of the regression to the true values of the often different set of correct structural parameters
# Argument:
#   regressors: A dictionary of dictionaries of dictionaries of regression objects for each parameter for each orphology
#   targets: a list of target morphologies
#   mapped_labs: the true morphology labels in the rearranged order
#   mapped_inds: a key mapping eah curve back to its unshuffled psoition
#   mapped_keys: a list of strings mapping each curve after shuffling back to its key in the database
#   dbd: a list of distances from the decision boundaries in the classifier
#   preds: the predicted morphologies of the curves
#   test_curves: the dictionary mapping each morphology to curves to be passed to the regression objects
#   test_params: the dictionary of dictionaries mapping each morphology to each stuctural parameters to the array of labels to use for the regression.
#   tmap: a map between the indices of the curves in the separate arrays to their concatenated counterparts
#   args: the argument object containing user defined flags
def compare_regression(regressors, targets, mapped_labs, mapped_inds, mapped_keys, dbd, preds, test_curves, test_params, tmap, args):
    for i in range(len(targets)):
        t = targets[i]
        examples = np.equal(mapped_labs, i)
        predicted = np.equal(preds, i)
        correct_inds = np.where(np.logical_and(examples, predicted))[0]
        incorrect_inds = np.where(np.logical_and(examples, np.logical_not(predicted)))[0]
        correct_file = open(join_path(args.resultsdir,'correct_%s.csv'%(targets[i])), 'w')
        correct_regs = {}
        original_correct = tmap[mapped_inds[correct_inds].astype(int)]
        correct_curves = test_curves[args.targets[i]][original_correct]
        for p in regressors[t].keys():
            correct_regs[p] = regressors[t][p].predict(correct_curves)
        for oci in range(len(original_correct)):
            oc = original_correct[oci]
            correct_file.write('%d TRUE %s REGRESSED %s\n'%(oc, ' '.join(['%s:%f'%(p, test_params[t][p][oc]) for p in test_params[t].keys()]), ' '.join(['%s:%f'%(p, correct_regs[p][oci]) for p in correct_regs.keys()])))
        correct_file.close()
        incorrect_file = open(join_path(args.resultsdir, 'incorrect_%s.txt'%(t)), 'w')
        original_incorrect = tmap[mapped_inds[incorrect_inds].astype(int)]
        incorrect_ck = mapped_keys[incorrect_inds]
        incorrect_curves = test_curves[args.targets[i]][original_incorrect]
        for ici in range(incorrect_inds.shape[0]):
            regression = {}
            ptarg = args.targets[preds[incorrect_inds[ici]].astype(int)]
            for p in regressors[ptarg].keys():
                regression[p] = regressors[ptarg][p].predict(np.reshape(incorrect_curves[ici], (1,-1)))
            incorrect_file.write('%d TRUE %s REGRESSED %s %s %s %s\n'%(original_incorrect[ici], ' '.join(['%s:%f'%(p, test_params[t][p][original_incorrect[ici]]) for p in test_params[t].keys()]), ptarg, ' '.join(['%s:%f'%(p, regression[p]) for p in regression.keys()]), incorrect_ck[ici], ' '.join(['%0.4f'%(dis) for dis in dbd[incorrect_ck[ici]]])))
    return

#evaluate regression
# This function simply evaluates the appropriate regression objects for each curve in a list of unlabeled curvs
# Arguents:
#   curves: an array of curves to be evaluated
#   classes: the list of predicted classes for thos curves
#   regressors: the dictionary of dictionaries mapping morphologies to structural parameters to regresoin objects
#   args: the argument object containing user specified parameters
# Returns:
#   predictions: a list of dictionaries mapping each curves to a prediction for each structural parameter
def evaluate_regression(curves, classes, regressors, args):
    predictions = []
    for i in range(curves.shape[0]):
        t = classes[i]
        regs = regressors[t]
        c = curves[i].reshape(1,-1)
        predictions += [{p: regs[p].predict(c) for p in regs.keys()}]
    return(predictions)

def evaluate_UQ(curves, classes, uncertainty_regressors, args):
    predictions = []
    mins = []
    maxs = []
    for i in range(curves.shape[0]):
        t = classes[i]
        regs = uncertainty_regressors[t]
        c = curves[i].reshape(1,-1)
        preds = {}
        temp_mins = {}
        temp_maxs = {}
        for p in regs.keys():
            preds[p], uq_bounds = regs[p].predict(c, alpha = 0.1)
            temp_mins[p] = uq_bounds[0,0,0]
            temp_maxs[p] = uq_bounds[0,1,0]
        predictions += [preds]
        mins += [temp_mins]
        maxs += [temp_maxs]
    return(predictions, mins, maxs)


# REORDER this helper function sets a list back to the original order, because the way the hierarchical model shifts the order.
# this is a stop gap in v 1.0 I've already smoothed this over and made it easier to use. and the user never needs to acknowledge the ordering or reordering.
# Arguments:
#   vals: the values to be unshuffles
#   mapped_inds: the originalindices shuffled the same 
def reorder(vals, mapped_inds):
    return(vals[np.argsort(mapped_inds)])

#read params
# returns a list of all unique structural parameters in all regression objects
def read_params(regressors):
    params = []
    for _, pdict in regressors.items():
        for key in pdict.keys():
            if key not in params:
                params += [key]
    return(params)


def main(args):
    #read in arguments
    #args = parse_args()
    targets = args.targets
    #load all data and instantiate regrssdion objects
    qs = np.loadtxt(join_path(args.datadir, 'q_200.txt'),dtype=str)
    q = loaders.load_q(args.datadir)
    train_curves = loaders.load_all_curves(targets, q, args.datadir, quotient = args.quotient)
    test_curves = loaders.load_all_curves(args.targets, q, args.datadir, prefix = 'TEST', quotient = args.quotient)
    gamma_norm = train_curves[args.targets[0]].shape[1]
    regressors = construct_regressor(join_path(args.configdir, args.reg_file), gamma_norm)
    param_list = read_params(regressors)
    extrap_params = param_list + [p for p in ['aspect_ratio', 'shell_ratio'] if p not in param_list]
    train_params = loaders.load_all_params(targets, extrap_params, args.datadir)
    test_params = loaders.load_all_params(targets, extrap_params, args.datadir, prefix = 'TEST')
    if args.extrapolation:
        test_curves, test_params = loaders.extrapolation_only(test_curves, test_params)

    # construct hierarchical struture
    ssf = open(join_path(args.configdir, args.hierarchy_file), 'r')
    struct_strings = ssf.readline().split()
    temp_ck_dict = loaders.load_all_params(args.targets, ['candidate key'], args.datadir, prefix='TEST')
    ck_dict = {t : temp_ck_dict[t]['candidate key'] for t in args.targets}
    ck, _ = loaders.concatenate_curves(ck_dict)
    curves, labels, _ = loaders.unravel_dict(train_curves, args.targets)
    print("CURVES %d %d"%(curves.shape[0], curves.shape[1]))
    tcurves, tlabels, tmap = loaders.unravel_dict(test_curves, args.targets)

    #train regression
    start_train = time.time()
    regressors = train_all_regression(train_curves, train_params, regressors)
    #reformat data insot single long arrays
    #instantiate and trainhierarchical model
    classifiers = hier.create_classifiers(struct_strings, gamma_norm)
    hierarchical_map, decisions = hierarchical_definition()
    hierarchical = hier.create_hierarchical(classifiers, decisions, curves, labels)
    #predict classification of test data and output classification report
    start_eval = time.time()
    preds, mapped_labs, mapped_inds, mapped_keys, dbd = hier.eval_hierarchical(classifiers, hierarchical_map, tcurves, tlabels, ck, True)
    end_eval = time.time()
    print("Training complete - training took %0.2f s - evaluation took %0.2f s Accuracy %f"%(start_eval - start_train, end_eval - start_eval, AS(tlabels, preds[np.argsort(mapped_inds)])))
    print(CR(tlabels, preds[np.argsort(mapped_inds)]))
    resfile = open(join_path(args.resultsdir, 'classification_results.txt'), 'w')
    resfile.write(CR(tlabels, preds[np.argsort(mapped_inds)]))
    #pass test data through regression objects and save data
    compare_regression(regressors, targets, mapped_labs, mapped_inds, mapped_keys, dbd, preds, test_curves, test_params, tmap, args)

    # if testing on other unlabeled data, read, classify, regress, and report
    if args.evaluate_file is not None:
        ecurves = loaders.load_txt(args.evaluate_file)
        eck = np.array(['eval%d'%(i) for i in range(curves.shape[0])])
        elabels = np.zeros(curves.shape[0])
        epreds, _, emapped_inds, emapped_keys, _ = hier.eval_hierarchical(classifiers, hierarchical_map, ecurves, elabels, eck, False)
        epreds = [targets[int(i)] for i in reorder(epreds, emapped_inds)]
        ekeys = reorder(eck, emapped_inds)
        evalfile = open(join_path(args.resultsdir, 'predictions.txt'), 'w')
        if args.uncertainty is False:
           predpars = evaluate_regression(ecurves, epreds, regressors, args)
           for i in range(len(epreds)):
              p = predpars[i]
              evalfile.write('%s %s\n'%(epreds[i], ' '.join(['%s:%s'%(k, p[k][0]) for k in p.keys()])))
        else:
           cal_curves = loaders.load_all_curves(targets, q, args.datadir, quotient = args.quotient, prefix = 'CALIBRATE')
           cal_params = loaders.load_all_params(targets, extrap_params, args.datadir, prefix = 'CALIBRATE')
           uq = calibrate_regression_UQ(regressors, cal_curves, cal_params)
           predpars, mins, maxs = evaluate_UQ(ecurves, epreds, uq, args)
           for i in range(len(epreds)):
              p = predpars[i]
              m = mins[i]
              M = maxs[i]
              print(m)
              print(p)
              print(M)
              evalfile.write('%s %s\n'%(epreds[i], ' '.join(['%s:[%s < %s < %s]'%(k, m[k], p[k][0], M[k]) for k in p.keys()])))


           #mapie.fit(X_cal, y_cal.reshape(-1,1))
           #mapie = MapieRegressor(estimator, test_size = 0.2, method = "plus", cv=10)
           #pred, pred_pis = mapie.predict(X_test, alpha = 0.05)
           
        



if __name__ == '__main__':
    args = parse_args()
    main(args)
