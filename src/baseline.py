import argparse
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold as KFold

import loaders

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--targets', default=['cylinder', 'disk', 'sphere', 'cs_cylinder', 'cs_disk', 'cs_sphere'], nargs = '+')
   parser.add_argument('--classifier', default='svc')
   parser.add_argument('--datadir', default='../example_data')
   parser.add_argument('--kernel', default='rbf')
   parser.add_argument('--degree', default=5, type=int)
   parser.add_argument('--c', default=100., help = 'The C parameter for SVC', type=float)
   parser.add_argument('--gamma', default=10., help='weight for SVC, ignored fo KNN and RF',type = float)
   parser.add_argument('--weight', default='uniform', help='The weight parameter for KNN, ignored otherwise')
   parser.add_argument('--n_neighbors', default=1, help='The number of neighbors for KNN, ignored otherwise', type=int)
   parser.add_argument('--logfile', default = None)
   parser.add_argument('--coeff0', default = 0.0, type = float)
   parser.add_argument('--n_est', default = 10000, type=int, help = 'The number of trees in a random forest')
   parser.add_argument('--rfcriterion', default='gini', type=str, help = 'The criterion to use for fandom forest')
   parser.add_argument('--max_depth', type=int, default=10, help = 'The max depth of trees in the random forest')
   parser.add_argument('--min_samples', default = 10, type=int, help = 'the minimum number of samples per split used in random forest')
   parser.add_argument('--max_features', default='sqrt', type=str, help = 'the maximum number of features to split on used in random forest')
   parser.add_argument('--k_fold', type=int, default=5, help = "The number of folds to use in K-fold cross validation, note for each fold one fold is used for training and the rest for validation")
   parser.add_argument('--train_folds', type=int, default = 1, help = "the number of folds to use as training, with the rest validation")
   parser.add_argument('--test', type=bool, default=False, help="a flag for whether to predict on the test data, if false will evaluate k-fold and report performance on validation")
   return(parser.parse_args())

def main():
   args = parse_args() #load arguments
   q = loaders.load_q(args.datadir) #this loads the q values of the curves, as strings to index into pandas DFs
   targets = args.targets
   train_curves = loaders.load_all_curves(targets, q, args.datadir)
   curves, labels, _ = loaders.unravel_dict(train_curves, args.targets)
   gamma_norm = args.gamma/curves[0].shape[0]

   #This next section selects which classifier to use
   #It then generates a text line containing the hyperpa5rameters to save to the logfile for future use
   if args.classifier == 'svc': #These next few lines construct an output string to write the classifier and paramters used to log
      predictor = SVC(C = args.c, gamma = gamma_norm, degree = args.degree, kernel = args.kernel, coef0 = args.coeff0, max_iter=50000000)
      outkey = 'SVC %f %f %f %d %s '%(args.c, args.gamma, args.coeff0, args.degree, args.kernel)
   elif args.classifier == 'knn':
      predictor = KNeighborsClassifier(n_neighbors = args.n_neighbors)
      outkey = 'KNN %d %s '%(args.n_neighbors, args.weight)
   elif args.classifier == 'random-forest':
      predictor = RandomForestClassifier(n_estimators = args.n_est, criterion = args.rfcriterion, max_depth = args.max_depth, min_samples_split = args.min_samples)
      outkey = 'RF %d %s %d %s %s '%(args.n_est, args.rfcriterion, args.max_depth, str(args.min_samples), str(args.max_features))
   else:
       print('Please specify a vlid classifier in: "svc", "knn", or "random-forest".')
       system.exit(1)

   #The section reads in a fully separate test set, trains the predictor on all available training data and then calculates the accuracy on the test set. All classes have equal representation in the test set so accuracy is a valid metric to use.
   if args.test is True:
      test_curves = loaders.load_all_curves(args.targets, q, args.datadir, prefix = 'TEST')
      tcurves, tlabels, tmap = loaders.unravel_dict(test_curves, args.targets)
      predictor.fit(curves, labels)
      predictions = predictor.predict(tcurves)
      print(predictor)
      print("%s ACCURACY: %0.3f"%(outkey, accuracy_score(tlabels, predictions)))

   else:
      #THIS is a modified k_fold that utilizes more data for validation than training. This helps ensure that a simppler model that is more extrapolable is selected as opposed to the risked everfitting with large amounts of training data and small amounts of validation data on which to evaluate it
      assert args.k_fold is not None, "Error plese provide either --k_fold argument or set --test to true"
      acc_vec = np.zeros(args.k_fold)
      skf = KFold(n_splits = args.k_fold)
      for i, (val_inds, train_inds) in enumerate(skf.split(curves, labels)): #train inds and val inds are purposefully swapped relative to the documentation
          predictor.fit(curves[train_inds], labels[train_inds])
          preds = predictor.predict(curves[val_inds])
          acc_vec[i] = accuracy_score(labels[val_inds], preds)
      print("Average accuracy: %0.3f"%(np.mean(acc_vec)))

      


if __name__ == '__main__':
    main()
