import numpy as np

alpha_range = np.arange(-2, 3)
gamma_range = np.arange(-1,3)
degree_range = np.arange(2,5)
coef0_range = [0,1,5]
outfile = open("regression_hyperparameters.txt", "w")
###RBF_strings
for alpha in alpha_range:
    for gamma in gamma_range:
        for coef0 in coef0_range:
            outfile.write("--alpha %f --gamma %f --kernel rbf --coef0 %f\n"%(10.**alpha, 10.**gamma, coef0))
            for degree in degree_range:
                outfile.write("--alpha %f --gamma %f --kernel polynomial --degree %d --coef0 %f\n"%(10.**alpha, 10.**gamma, degree, coef0))
outfile.close()

