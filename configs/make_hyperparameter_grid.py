import numpy as np

C_range = np.arange(-2, 3)
gamma_range = np.arange(-1,3)
degree_range = np.arange(2,5)
coef0_range = [0,1]
outfile = open("hyperparameters.txt", "w")
###RBF_strings
for c in C_range:
    for gamma in gamma_range:
        for coef0 in coef0_range:
            outfile.write("svc_%f_%f_rbf_%f\n"%(10.**c, 10.**gamma, coef0))
            for degree in degree_range:
                outfile.write("svc_%f_%f_poly_%d_%f\n"%(10.**c, 10.**gamma, degree, coef0))
outfile.close()

