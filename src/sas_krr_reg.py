import os
import sys
import numpy as np
from sklearn.decomposition import PCA

def construct_manifold(spec, num_comps):
   transformer = PCA(n_components = int(num_comps)).fit(spec)
   return(transformer)

def construct_regression(X, y, alpha, gamma_0, kernel = 'polynomial', degree=5):
   gamma = gamma_0/X.shape[0]
   if kernel == 'polynomial':
      regression = KRR(kernel = kernel, gamma = gamma, alpha = alpha, degree=degree).fit(X,y)
   elif kernel == 'linear':
      regression = KRR(kernel = kernel, gamma = gamma, alpha = alpha).fit(X,y)
   return(regression)

def map_all_pcas(spec_dict, manifolds, num_comps):
   transformers = {}
   print('MANIFOLDS')
   print(manifolds)
   for i in range(len(manifolds)):
      manifold = manifolds[i]
      print(manifold)
      transformer = construct_manifold(np.concatenate([spec_dict[t] for t in manifold]), num_comps[i])
      for t in manifold:
         transformers[t] = transformer
   return(transformers)


      
   

