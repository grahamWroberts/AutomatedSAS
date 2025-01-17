import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_curves(fn, q):
   if type(fn) == str:
    indf = pd.DataFrame(pd.read_csv(fn))
   else:
    indf = fn
   curves = scale_highq(np.log10(np.array(indf.loc[:,q])+1),0.001)
   #curves = np.log10(np.array(indf.loc[:,q])+0.001)
   return(curves)

def load_params(fn, colnames):
   if type(fn) == str:
    indf = pd.DataFrame(pd.read_csv(fn))
   else:
    indf = fn
   outparams = {}
   #indf = pd.DataFrame(pd.read_csv(fn))
   for cn in colnames:
      if cn in indf.columns:
         outparams[cn] = np.array(indf.loc[:,cn])
      else:
         outparams[cn] = np.zeros(len(indf))
   return(outparams)

def scale(curves, maxval):
   for i in range(curves.shape[0]):
      curves[i,:] = curves[i,:]-curves[i,0]+maxval
   return(curves)

def scale_highq(curves, incoherence):
   new_curves = curves.copy()
   for i in range(curves.shape[0]):
       new_curves[i] = curves[i] -np.mean(curves[i,-2:])+incoherence
   return(new_curves)

def load_q(datadir, qfile = 'q_200.txt'):
   q = np.loadtxt('%s/%s'%(datadir, qfile), dtype=str, delimiter=',')
   return(q)

def load_all_curves(targets, q, datadir, prefix='TRAIN'):
   all_curve = {}
   maxval = 0
   for t in targets:
      fn = '%s/%s_%s.csv'%(datadir, prefix, t)
      df = pd.DataFrame(pd.read_csv(fn))
      curve = load_curves(df, q)
      all_curve[t] = curve
   return(all_curve)

def load_txt(filename, incoherrence = 0.001, blur = 1.0):
    infile = np.loadtxt(filename)
    curves = scale_highq(np.log10(infile+1),incoherrence)
    return(curves)

def load_params(fn, colnames):
   if type(fn) == str:
    indf = pd.DataFrame(pd.read_csv(fn))
   else:
    indf = fn
   outparams = {}
   #indf = pd.DataFrame(pd.read_csv(fn))
   for cn in colnames:
      if cn in indf.columns:
         outparams[cn] = np.array(indf.loc[:,cn])
      else:
         outparams[cn] = np.zeros(len(indf))
   return(outparams)

def load_all_params(targets, ps, datadir, prefix='TRAIN'):
   all_curve = {}
   maxval = 0
   for t in targets:
      all_curve[t] = {}
      fn = '%s/%s_%s.csv'%(datadir, prefix, t)
      df = pd.DataFrame(pd.read_csv(fn))
      curve = load_params(df, ps)
      all_curve[t] = curve
   return(all_curve)

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

def extrapolation_only(curves, params, aspect_ratio_cutoff = 8, shell_ratio_cutoff = 0.65):
    newcurves = {}
    newparams = {}
    for t in curves.keys():
        ar = params[t]["aspect_ratio"]
        sr = params[t]["shell_ratio"]
        extrapolate_aspect_ratio = np.logical_or(np.greater(ar, aspect_ratio_cutoff), np.less_equal(ar, 1./aspect_Ratio_curoff))
        extrapolate_shell = np.greter_equal(sr, shell_ratio_cutoff)
        valid = np.where(np.logical_and(extrapolate_aspect_ratio, extrapolate_shell))[0]
        newcurves[t] = curves[t][valid]
        newparams[t] = {params[t][p][valid] for p in params[t].keys()}
    return
