#AutomatedSAS
#Author Graham Roberts

#loaders.py a set of functions for loading and preprocessing the data
import numpy as np
import pandas as pd
from os.path import join as join_path

# quotient transform
# This is a technique employed in Yildirim "SASFormer"
# S(q_i) = I(q_i)/I(q_{i+1})
def quotient_transform(I):
    if I.ndim == 1:
        S = I[:-1]/I[1:]
    else:
        S = I[:,1:]/I[:,:-1]
    return(S)

# load curves
# reads a curve from a pandas dataframe csv and formats it
# Arguments:
#  fn: the filepath to the file to be read in
#  q: an a rray of q values of the curve in string form which ar the column header of the dataframe
# Returns:
#  curves: an array on [n_curves, n_features] of all the curves
def load_curves(fn, q, quotient = False):
   if type(fn) == str:
    indf = pd.DataFrame(pd.read_csv(fn))
   else:
    indf = fn
   if not quotient:
      #curves = scale_highq(np.log10(np.array(indf.loc[:,q])),0.001)
      curves = scale_highq(np.log10(np.array(indf.loc[:,q])+1),-3.)
   else:
       curves = quotient_transform(np.array(indf.loc[:,q]))
   #curves = np.log10(np.array(indf.loc[:,q])+0.001)
   return(curves)

#load params:
#loads a dataframe from file and reads parameters from it
#Argument:
#  fn: either the filepath to a dataframe to load or the dataframe itself
#  colnames: a list of columns to read
#Returns: 
#  outparams: a dictionary mapping each parameter to the array of values
def load_params(fn, colnames):
   if type(fn) == str:
    indf = pd.DataFrame(pd.read_csv(fn))
   else:
    indf = fn
   outparams = {}
   for cn in colnames:
      if cn in indf.columns:
         outparams[cn] = np.array(indf.loc[:,cn])
      else:
         outparams[cn] = np.zeros(len(indf))
   return(outparams)

#def scale(curves, maxval):
#   for i in range(curves.shape[0]):
#      curves[i,:] = curves[i,:]-curves[i,0]+maxval
#   return(curves)

#scale_highq
# scales a curve by shifting it in logspce to a prior defined incoherence value
# Arguments:
#   curves: an array of curves
#   incoherence: a new background value to shift the curves to
def scale_highq(curves, incoherence):
   new_curves = curves.copy()
   for i in range(curves.shape[0]):
       new_curves[i] = curves[i] -np.mean(curves[i,-10:])+incoherence
   return(new_curves)

#load_q
#just a helper function for how the q data are stores
def load_q(datadir, qfile = 'q_200.txt'):
   q = np.loadtxt(join_path(datadir, qfile), dtype=str, delimiter=',')
   return(q)

#load_all_curves
# Loads all curves by iterating through all target morphologies and formatting the filepath to the data and calling load_curves
# Arguments:
#   targets: a list of target morphologies
#   q: a list of q values as string to use as column names
#   datadir: the directory where data are stored
#   prefix: either TRAIN or TEST, which deliniates which dataset is to be loaded
# Return:
# all_curve: a dictionary mapping each morphology to an array
def load_all_curves(targets, q, datadir, prefix='TRAIN', quotient=False):
   all_curve = {}
   maxval = 0
   for t in targets:
      fn = join_path(datadir, '%s_%s.csv'%(prefix, t))
      df = pd.DataFrame(pd.read_csv(fn))
      curve = load_curves(df, q, quotient)
      all_curve[t] = curve
   return(all_curve)

#load_txt
# loads data in much the same way, but with no knowledge of morphology or parameters
# assumes data are saved in a txt or csv
# used to load experimental data that are saved in a simmpler format
# argument:
#   filename: the path to the file
#   incoherence: a now background to scale the data to to make it similar to simlated data with a background at incoherence
#   blur: a noise value which is added to the raw form of the curves
def load_txt(filename, incoherrence = -3, blur = 1.0):
    infile = np.loadtxt(filename)
#    infile[0,:] += 1e1
#    infile[6,:] += 1e-3
    #curves = np.log10(infile)
    curves = scale_highq(np.log10(infile),incoherrence)
    return(curves)


#load all params
#loads all structural paramerters for each morphology
# Arguments:
#   target: a list of target morphologies
#    ps: a list of structural parameters
#    datadir: the directory where the data are saved
#    prefic: either 'TRAIN' or 'TEST' deiniating whether to load the training or test data
# Returns:
#    all_curve a dicionary of dictionaries of arrays mapping each morphology to each structural parameter to an array of labels
def load_all_params(targets, ps, datadir, prefix='TRAIN'):
   all_curve = {}
   maxval = 0
   for t in targets:
      all_curve[t] = {}
      fn = join_path(datadir, '%s_%s.csv'%(prefix, t))
      df = pd.DataFrame(pd.read_csv(fn))
      curve = load_params(df, ps)
      all_curve[t] = curve
   return(all_curve)

#concatenate_curves
# takes a dictionary mapping morpholgoes to curves and concatenates all curves into one large array
# returns the large array and an array which target each curve belonged o as an enumerate integer
def concatenate_curves(curve):
   key = 0
   curve_list = []
   label_list = []
   for t in curve.keys():
      curve_list += [curve[t]]
      label_list += [key*np.ones(curve[t].shape[0])]
      key += 1
   allcurve = np.concatenate(curve_list)
   labels = np.concatenate(label_list)      
   return(allcurve, labels)

#unravel_dict
#creates a large arraay of curves from different classes
# can be used to create an array using a subset of classes
# returns the array of curves, the array of labels, and the original index of each curve
def unravel_dict(curve_dict, targets = None):
    if targets is None:
        targets = curve_dict.keys()
    out_curve = curve_dict[targets[0]]
    out_labels = np.zeros(out_curve.shape[0])
    out_map = np.arange(out_curve.shape[0])
    for i in range(1,len(targets)):
        t = targets[i]
        out_curve = np.concatenate((out_curve, curve_dict[t]))
        out_labels = np.concatenate((out_labels, i*np.ones(curve_dict[t].shape[0])))
        out_map = np.concatenate((out_map, np.arange(curve_dict[t].shape[0])))
    return(out_curve, out_labels, out_map)

#extrapolation only
#removel all cuves in the test set that dont have either aspect_ratio or shell_ratio outside a certain threshold, used to evaluate performance only on data with these values outside the range of training data
#Arguments:
#   curves: the dictionary of arrays of curves
#   params: the dictionary of dictionaries of arrays of structural parameter values for each curve, must include 'aspect_ratio', and 'shell_ratio', random values are used to maintian class balance with morphologoes that don't incude these
def extrapolation_only(curves, params, aspect_ratio_cutoff = 8, shell_ratio_cutoff = 0.65):
    newcurves = {}
    newparams = {}
    for t in curves.keys():
        ar = params[t]["aspect_ratio"]
        sr = params[t]["shell_ratio"]
        extrapolate_aspect_ratio = np.logical_or(np.greater(ar, aspect_ratio_cutoff), np.less_equal(ar, 1./aspect_ratio_cutoff))
        extrapolate_shell = np.greater_equal(sr, shell_ratio_cutoff)
        valid = np.where(np.logical_or(extrapolate_aspect_ratio, extrapolate_shell))[0]
        newcurves[t] = curves[t][valid]
        newparams[t] = {p:params[t][p][valid] for p in params[t].keys()}
    return(newcurves, newparams)

#remap external
#This function is for preprocessing experimental urves drawn from other sources
#It takes some curve mapped to arbitrary q values and interpolates/extrapolates onto the q values expected by the model.
#Note that when extrapolating it simply takes either the last value further out or the first value prior as curves tend to plateau in these regions. It is reccomended to extrapolate as little as possible.
#ARGS: q: the array of q values expected by the ML models
#  exp_q: The array of q values the external curve is already measured at
#  exp_I: The array of intensity values measured at the q values in exp_q
#RETURN: A new curve in log space estimated at the q values specified in the first argument
def remap_external(q, exp_q, exp_I):
    interim = np.where(np.logical_and(np.less_equal(q, np.max(exp_q)), np.greater_equal(q, np.min(exp_q))))[0]
    new_curve = np.zeros(q.shape)
    new_curve[interim] = np.interp(q[interim], exp_q, exp_I)
    if interim[-1] < len(q)-1:
        cutoff = interim[-1]
        new_curve[cutoff+1:] = new_curve[cutoff]
    if np.min(interim) > 0:
        cutoff = interim[0]
        new_curve[:cutoff] = new_curve[cutoff]
    scaled_curve = scale_highq(np.log10(new_curve), -3)
    return(new_curve)

