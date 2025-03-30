import numpy as np
import sys
sys.path.append("./src")
import argparse
import os
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import loaders
import time

#parse args
#reads in option flags when script is invoked
def parse_arguments():
   parser = argparse.ArgumentParser()
   parser.add_argument('--targets', default = ['cylinder', 'disk', 'sphere', 'cs_cylinder', 'cs_disk', 'cs_sphere'], nargs = '+')
   parser.add_argument('--params', default = ['radius', 'length', 'shell'])
   parser.add_argument('--datadir', default = './data', help = 'the directory where the raw data are stored')
   parser.add_argument('--configdir', default = '../configs', help = 'the directory where the configuration files for the classifier and regressor are stored')
   parser.add_argument('--resultsdir', default = '../results', help = 'the directory where results and logs will be stored')
   parser.add_argument('--quotient', type=bool, default=False)
   parser.add_argument('--hyperparameters', type=str, default = 'regression_hyperparameters.txt', help = "a file containing the regression hyperparameters to check")
   return(parser.parse_args())

map_of_params = {"cylinder": ["radius", "length"],
                 "disk": ["radius", "length"],
                 "sphere": ["radius"],
                 "cs_cylinder": ["radius", "length", "shell"],
                 "cs_disk": ["radius", "length", "shell"],
                 "cs_sphere": ["radius", "shell"]}
def remove_extra_params(params):
    new_params = {}
    for t in params.keys():
        new_params[t] = {}
        for p in map_of_params[t]:
            new_params[t][p] = params[t][p]
    return(new_params)

def create_regression_hyperparameters(lines):
    outlist = []
    for line in lines:
        tokens = line.split()
        pset = {}
        for i in range(len(tokens)):
            if "--" in tokens[i]:
                pname = tokens[i].replace("--","")
                pval = tokens[i+1]
                pset[pname] = pval
        outlist += [pset]
    return(outlist)
def main():
    args = parse_arguments()
    targets = args.targets
    hyperparameters = open(os.path.join(args.configdir, args.hyperparameters), 'r').readlines()
    hyperlist = create_regression_hyperparameters(hyperparameters)
    q = loaders.load_q(args.datadir)
    train_curves = loaders.load_all_curves(args.targets, q, args.datadir, quotient = args.quotient)
    outfile = open(os.path.join(args.configdir, "tuned_hyperparameters.txt"), "w")
    timefile = open("tuning_times.txt", "w")
    param_list = args.params
    extrap_params = param_list + [p for p in ['aspect_ratio', 'shell_ratio'] if p not in param_list]
    train_params = loaders.load_all_params(targets, extrap_params, args.datadir)
    train_params = remove_extra_params(train_params)
    print(hyperlist)
    outfile = open(os.path.join(args.configdir,"tuned_regression.txt"), "w")
    regs = {}
    for t in targets:
        regs[t] = {}
        for p in param_list:
            if p in train_params[t].keys():
                best_perf = 10000
                best_conf = None
                for conf in hyperlist:
                    if conf["kernel"] == "polynomial":
                        krr = KRR(kernel = "polynomial", degree = int(conf["degree"]), coef0 = float(conf["coef0"]), gamma = float(conf["gamma"])/200., alpha = float(conf["alpha"]))
                    else:
                        krr = KRR(kernel = "rbf", coef0 = float(conf["coef0"]), gamma = float(conf["gamma"])/200., alpha = float(conf["alpha"]))
                    perfs = np.zeros(5)
                    skf = KFold(n_splits=5)
                    for i, (val_inds, train_inds) in enumerate(skf.split(train_curves[t], train_params[t][p])):
                        krr.fit(train_curves[t][train_inds], train_params[t][p][train_inds])
                        preds = krr.predict(train_curves[t][val_inds])
                        perfs[i] = MAPE(train_params[t][p][val_inds], preds)
                    if np.mean(perfs) < best_perf:
                        best_perf = np.mean(perfs)
                        best_conf = conf
                outfile.write("--target %s --param %s %s\n"%(t, p, " ".join(["--%s %s"%(key, val) for key, val in best_conf.items()])))
    outfile.close()



if __name__ == "__main__":
    main()
