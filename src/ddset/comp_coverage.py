### compute data coverage of data set
import numpy as np
vio_max = 1e-8

def comp_coverage_ellip(X, SigmaInv2_list, beta_list, varrho_list):

    inside_num_ellip = 0
    K = len(SigmaInv2_list)
    n = len(X)

    vio_ind = []
    
    for _, point in enumerate(X):
        vio_max = 0
        vio_list = []

        for k in range(K):

            _vio = np.linalg.norm(SigmaInv2_list[k]@(point - beta_list[k]), 2) - varrho_list[k] 
            vio_list.append(_vio)

        vio = min(vio_list)
        if vio <= vio_max:
            inside_num_ellip += 1
            vio_list.append(0)
        else:
            vio_list.append(1)
            
    coverage_ellip = inside_num_ellip/n
    return coverage_ellip, vio_ind


def comp_coverage_box(X, D_list, d_list):
    
    inside_num_box = 0
    n = X.shape[0]
    K = len(D_list)
    vio_ind = []
    for _, point in enumerate(X):
        vio_max = 1e-2
        vio_list = []

        for k in range(K):
            _vio = np.max(D_list[k]@point - d_list[k])
            vio_list.append(_vio)

        vio = min(vio_list)
        if vio <= vio_max:
            inside_num_box += 1
            vio_ind.append(0)
        else:
            vio_ind.append(1)

    coverage_box = inside_num_box/n
    return coverage_box, vio_ind