import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
import pandas as pd
import os
import sys
sys.path.append('..')
sys.path.append('../krr')
import loaders
import sas_krr_reg as spreg
from sklearn.svm import SVC
from sklearn.metrics import classification_report as CR
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import confusion_matrix as CM
from matplotlib import pyplot as plt

def relabel(labels, decision):
   new_inds = np.empty(0)
   new_labels = np.empty(0)
   for i in decision.keys():
      temp_matches = np.equal(labels, i)
      temp_labels = decision[i]*np.ones(np.sum(temp_matches))
      temp_inds = np.where(temp_matches)[0]
      new_inds = np.concatenate((new_inds, temp_inds))
      new_labels = np.concatenate((new_labels, temp_labels))
   return(new_inds.astype(int), new_labels)

def create_classifiers(struct_strings, gamma_norm):
   classifiers = []
   for ss in struct_strings:
      ss = ss.strip()
      toks = ss.split('_')
      print(toks)
      if toks[0].lower() == 'svc':
          if toks[3].lower()== 'poly':
            classifier = SVC(C = float(toks[1]), gamma = float(toks[2])/gamma_norm, kernel = 'poly', degree = int(toks[4]), coef0=float(toks[5]), max_iter = 500000)
          else:
            classifier = SVC(C = float(toks[1]), gamma = float(toks[2])/gamma_norm, kernel = toks[3], coef0 = float(toks[4]), max_iter = 500000)
      #elif toks[0].lower() == 'knr':
         #classifier = 
      classifiers += [classifier]
   return(classifiers)
   
def create_hierarchical(classifiers, decisions, train_spec, train_labels):
   for i in range(len(classifiers)):
      temp_inds, temp_labels = relabel(train_labels, decisions[i])
      classifiers[i].fit(train_spec[temp_inds], temp_labels)
   return()

def eval_hierarchical(classifiers, hierarchical_map, spec, labels, inck = None, db_dist = False):
   if inck is None:
       ck = np.zeros(spec.shape[0])
   else:
       ck = inck
   dbd = False
   if db_dist == True and inck is not None:
       dbd = True
       db_dists = {}
   else:
       db_dists = None
   inds = np.arange(spec.shape[0])
   queue = [0]
   spec_queue = [spec]
   key_queue = [labels]
   inds_queue = [inds]
   ck_queue = [ck]
   out_preds = np.empty(0)
   out_key = np.empty(0)
   out_inds = np.empty(0)
   out_ck = np.empty(0)
   while(len(queue)>0):
      node = queue[0]
      temp_spec = spec_queue[0]
      temp_key = key_queue[0]
      temp_inds = inds_queue[0]
      temp_ck = ck_queue[0]
      temp_preds = classifiers[node].predict(spec_queue[0])
      if dbd:
         #w = np.linalg.norm(classifiers[node].coef_)
         temp_d = classifiers[node].decision_function(spec_queue[0])
         for i in range(temp_d.shape[0]):
             key = ck_queue[0][i]
             if key in db_dists.keys():
                 db_dists[key] += [temp_d[i]]
             else:
                 db_dists[key] = [temp_d[i]]
      for bin_val in [0,1]:
         match_inds = np.where(np.equal(temp_preds, bin_val))[0]
         if type(hierarchical_map[node][bin_val]) == int:
            if match_inds.shape[0] > 0:
               queue += [hierarchical_map[node][bin_val]]
               spec_queue += [temp_spec[match_inds]]
               key_queue += [temp_key[match_inds]]
               inds_queue += [temp_inds[match_inds]]
               ck_queue += [temp_ck[match_inds]]

         else:
            out_preds = np.concatenate((out_preds, int(hierarchical_map[node][bin_val])*np.ones(match_inds.shape[0])))
            out_key = np.concatenate((out_key, temp_key[match_inds]))
            out_inds = np.concatenate((out_inds, temp_inds[match_inds]))
            out_ck = np.concatenate((out_ck, temp_ck[match_inds]))
      queue = queue[1:]
      spec_queue = spec_queue[1:]
      key_queue = key_queue[1:]
      inds_queue = inds_queue[1:]
      ck_queue = ck_queue[1:]
   return(out_preds, out_key, out_inds, out_ck, db_dists)
            
