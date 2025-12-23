###  the test example is for reproducing the results shown in the paper: Li, Y., Yorke-Smith, N., & Keviczky, T. (2024). Machine learning enabled uncertainty set for data-driven robust optimization. Journal of Process Control, 144, 103339.

import numpy as np
from sklearn import preprocessing
import ddset


X_orig = np.load("./tests/train_data_raw.npy")
X_train = X_orig[:,:2]

scaler = preprocessing.MaxAbsScaler().fit(X_train)
X_scaled = scaler.transform(X_train)

eps = 0.0115
MinPts = 3
K = 6

dgp = ddset.DGP()
dgp.fit(X_scaled, eps, MinPts, K, if_box = True, if_ellip = True)

print("the outlier percentage is: {0}%".format(dgp.outlier_pct))
print("the data coverage of the box uncertainty set is: {0}".format(dgp.coverage_ellip))

dgp.visualization()