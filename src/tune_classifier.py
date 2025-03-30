import numpy as np
import sys
sys.path.append("./src")
import argparse
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report as CR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold as KFold
import loaders
import time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default = "data")
    parser.add_argument("--configdir", default = "configs")
    parser.add_argument("--decisions", default = "decisions.txt")
    parser.add_argument("--hyperparameters", default="hyperparameters.txt")
    parser.add_argument("--quotient", default=False)
    parser.add_argument("--targets", default = ['cylinder', 'disk', 'sphere', 'cs_cylinder', 'cs_disk', 'cs_sphere'])
    parser.add_argument("--k_fold", default=5)
    return(parser.parse_args())

def make_classifier(structure):
    tokens = structure.replace("\n","").split("_")
    gamma_norm = 200
    if tokens[3] == "rbf":
        classifier = SVC(kernel='rbf', C=float(tokens[1]), gamma = float(tokens[2])/gamma_norm, coef0 = int(float(tokens[4])))
    else:
        classifier = SVC(C=float(tokens[1]), gamma = float(tokens[2])/gamma_norm, kernel=tokens[3], degree = int(float(tokens[4])), coef0 = int(float(tokens[5])))
    return(classifier)

def separate_data(decision, train_data):
    classes = decision.replace("\n","").split("<->")
    class1 = np.concatenate([train_data[t] for t in classes[0].split()])
    lab1 = np.zeros(class1.shape[0])
    class2 = np.concatenate([train_data[t] for t in classes[1].split()])
    lab2 = np.ones(class2.shape[0])
    temp_data = np.concatenate([class1, class2])
    temp_labels = np.concatenate([lab1, lab2])
    return(temp_data, temp_labels)

def evaluate_classifier(classifier, data, labs, k_fold):
    skf = KFold(n_splits=k_fold)
    perfs = np.zeros(k_fold)
    for i, (val_inds, train_inds) in enumerate(skf.split(data, labs)):
       classifier.fit(data[train_inds], labs[train_inds])
       preds = classifier.predict(data[val_inds])
       perfs[i] = accuracy_score(labs[val_inds], preds)
    return(np.mean(perfs))


def main():
    args = parse_arguments()
    decisions = open(os.path.join(args.configdir, args.decisions), 'r').readlines()
    hyperparameters = open(os.path.join(args.configdir, args.hyperparameters), 'r').readlines()
    q = loaders.load_q(args.datadir, qfile = 'q_yldirim.txt')
    train_curves = loaders.load_all_curves(args.targets, q, args.datadir, quotient = args.quotient, prefix='YLDIRIM_TRAIN')
    outfile = open(os.path.join(args.configdir, "tuned_hyperparameters.txt"), "w")
    timefile = open("tuning_times.txt", "w")
    for decision in decisions:
        temp_data, temp_labels = separate_data(decision, train_curves)
        best_perf = 0
        best = None
        for option in hyperparameters:
           print(option)
           start_time = time.time()
           classifier = make_classifier(option)
           perf = evaluate_classifier(classifier, temp_data, temp_labels, args.k_fold)
           if perf > best_perf:
              print("%f > %f"%(perf, best_perf))
              best_perf = perf
              best = option
           end_time = time.time()
           print(end_time - start_time)
           timefile.write("%f\n"%(end_time - start_time))
        outfile.write("%s "%(best.replace("\n","")))



main()
