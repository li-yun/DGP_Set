#####################################################
###  this documents contains all basic components for generating DPG-Set
#####################################################


import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn import cluster

#from sklearn import preprocessing

import alphashape
from shapely.geometry import Polygon, Point, LineString

#from descartes import PolygonPatch
import alphashape
import shapely
from sklearn.mixture import GaussianMixture

import time

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from .comp_coverage import comp_coverage_ellip, comp_coverage_box
from .visual import plot_set


class DGP():
    def __init__(self):  ### eps: maximum distance between two samples; MinPts: minimal number of samples a point needs to have; K: number of clusters; if_ellipsoid: where generate ellipsoidal uncertainty set; if_box: whether to geneterate box uncertainty set
        
        self.X = None # raw_data_set
        self.m = None # number of features
        self.n = None # number of samples
        self.eps = None # epsilon for DBSCAN
        self.MinPts = None # Minimal points for DBSCAN
        self.K = None # (Int), number of clusters for GMM
        self.if_ellipsoid = None # (Boolean), whether to generate ellipsoidal uncertainty subsets
        self.if_box = None # (Boolean), whether to generate box uncertainty subsets
        self.coverage_ellip = None # data coverage of ellip uncertainty set
        self.coverage_box = None # data coverage of box uncertainty set
        self.outlier_pct = None # outlier percentage
        self.X_list = [] # data samples list for each cluster
        self.outliers = None # data samples identified as outliers
        self.score = None # the score for measuring the performance of clustering (depends on the chosen score, such as Sihouette score)
        self.D_list = [] # list of generated polyhedral uncertainty set params: Dw <= d
        self.d_list = [] # list of generated polyhedral uncertainty set params: Dw <= d
        self.beta_list = [] # list of generated ellipsoidal uncertainty set params:
        self.SigmaInv2_list = [] # list of generated ellipsoidal uncertainty set params: ||SigmaInv2(w - beta)||_2 <= varho
        self.varrho_list = [] # list of generated ellipsoidal uncertainty set params

        self.vio_ind_ellip = [] # index showing whether the corresponding uncertainty sample is in or out the uncertainty set
        self.vio_ind_box = []
        
        self.shp_list_box = [] # parameter list of alphashape
        self.shp_list_ellip = [] # parameter list of alphashape

        ## test set data coverage and index (ind = 0, sample is in uncertainty set, otherwise, out of uncertainty set)
        self.test_pct_box = None
        self.test_ind_box = None
        self.test_pct_ellip = None
        self.test_ind_ellip = None

    def reset(self):  # for each training batch, reset the parameters related to the uncertainty sets
        self.shp_list_box = []
        self.shp_list_ellip = []
        self.vio_ind_ellip = []
        self.vio_ind_box = []
        self.X_list = []
        self.beta_list = []
        self.SigmaInv2_list = []
        self.varrho_list = []
        self.D_list = []
        self.d_list = []

    def fit(self, X, eps: float, MinPts: int, K: int, if_ellip = False, if_box = True, verbose = True):
        self.X = X
        self.m = X.shape[1]
        self.n = X.shape[0]

        self.eps = eps
        self.MinPts = MinPts
        self.K = K

        self.if_ellipsoid = if_ellip
        self.if_box = if_box

        self.reset()

        ### implement DBSCAN to remove outliers
        dbscan = cluster.DBSCAN(eps = self.eps, min_samples = self.MinPts).fit(self.X)
        X_clean = self.X[dbscan.labels_ != -1, :]
        outliers = self.X[dbscan.labels_ == -1, :]
        self.outliers = outliers

        self.outlier_pct = len(outliers)/self.n * 100  # compute outlier percentage

        
        ### implement GMM to generate data clusters
        gmm = GaussianMixture(n_components = self.K, covariance_type = "full", n_init = 100)
        gmm.fit(X_clean)
        cls_labels = gmm.predict(X_clean)
        self.score = silhouette_score(X_clean, cls_labels) 

        for i in range(self.K):
            self.X_list.append(X_clean[cls_labels == i,:])

        ### generate ellipsoidal uncertainty sets
        if self.if_ellipsoid:
            for k in range(self.K):
                X = self.X_list[k]
                X_mean = gmm.means_[k,:]
                Sigma = gmm.covariances_[k]
                Sigma_inv2 =  comput_sqrtM(Sigma) #compute the -1/2 of the covariance matrix
                varrho = np.max(np.array([np.linalg.norm(Sigma_inv2@(X[t,:] - X_mean), 2) for t in range(X.shape[0])])) 
        
                self.beta_list.append(X_mean)
                self.SigmaInv2_list.append(Sigma_inv2)
                self.varrho_list.append(varrho)

                ### if the uncertainty sample has two features, the following alphashapes are generated for visualization
                if self.m == 2:
                    X_centered = X - X_mean
                    pc1_range = np.linspace(np.min(X_centered[:,0])*(3),np.max(X_centered[:,0])*(3),500) + X_mean[0]
                    pc2_range = np.linspace(np.min(X_centered[:,1])*(3),np.max(X_centered[:,1])*(3),500) + X_mean[1]
                    
                    Points_ellip = []
                    
                    for k in range(len(pc1_range)):
                        for j in range(len(pc2_range)):
                            point = np.array([pc1_range[k], pc2_range[j]])
                            if np.linalg.norm(Sigma_inv2@(point - X_mean), 2) <= varrho:
                                Points_ellip.append(point)
                            
                    shp = alphashape.alphashape(Points_ellip, alpha = 0.)
                    self.shp_list_ellip.append(shp)            

            # ### save the data generated
            # if if_save:
            #     np.savez("ellipsoid.npz", beta_list = self.beta_list, SigmaInv2_list = self.SigmaInv2_list, varrho_list = self.varrho_list)


            self.coverage_ellip, self.vio_ind_ellip = comp_coverage_ellip(self.X, self.SigmaInv2_list, self.beta_list, self.varrho_list)  # compute data coverage

            if verbose:
                print("\n"+"-"*20 + "training results for ellipsoidal uncertainty subsets" + "-"*20)
                print("-- number of uncertainty subsets: {0}".format(self.K))
                print("-- Silhouette score: {0:.4f}".format(self.score))
                print("-- data coverage of training data: {0:.4f}".format(self.coverage_ellip))
            
            

        ### generate box uncertainty set
        if self.if_box:
            for i, X in enumerate(self.X_list):
    
                if len(X) == 0:
                    continue                     
                X_mean = np.mean(X, axis = 0)
                X_centered = X - X_mean 
                
                pca = PCA(n_components = self.m).fit(X_centered)
                
                X_pca = pca.fit_transform(X_centered)
                
                pca_min = np.min(X_pca,axis = 0)
                pca_max = np.max(X_pca,axis = 0)
    

                ### if feature dimension is 2, generate visualization results ###
                if self.m == 2:
                    X_min = pca.components_@pca_min
                    X_max = pca.components_@pca_max
                    pca1_minmax = [pca_min[0], pca_max[0]]
                    pca2_minmax = [pca_min[1], pca_max[1]]

                    ext_points = [] ### extreme points 
                    for pca_1 in pca1_minmax:
                        for pca_2 in pca2_minmax:
                            ext_points.append([pca_1,pca_2])
                            
                    alpha_points_list = []
                    for point in ext_points:
                        alpha_points_list.append(X_mean + pca.components_.T@np.array(point))
                        
                    shp = alphashape.alphashape(alpha_points_list, alpha = 0.)
                    
                    self.shp_list_box.append(shp)
                    
                P = pca.components_.T
            
                self.D_list.append(np.kron(np.array([[1],[-1]]), P.T))
                self.d_list.append(np.concatenate((pca_max+P.T@X_mean, - pca_min - P.T@X_mean)))


            self.coverage_box, self.vio_ind_box = comp_coverage_box(self.X, self.D_list, self.d_list)  # compute data coverage
            
            if verbose:
                print("\n"+"-"*20 + "training results for box uncertainty subsets" + "-"*20)
                print("-- number of uncertainty subsets: {0}".format(self.K))
                print("-- Silhouette score: {0:.4f}".format(self.score))
                print("-- data coverage of training data: {0:.4f}".format(self.coverage_box))
            

    def save_sets(self): # save the uncertainty set parameters
        if self.if_box:
            np.savez("box.npz", D_list = self.D_list, d_list = self.d_list)

        if self.if_ellipsoid:
            np.savez("ellipsoid.npz", beta_list = self.beta_list, SigmaInv2_list = self.SigmaInv2_list, varrho_list = self.varrho_list)

    
    def visualization(self, save_fig = False, file_names = ["",""]): # visualize the uncertainty sets and training data
        pos_params_plot = (self.m, self.outliers, self.X_list)
        name_params_plot = {"shp_list_box":self.shp_list_box, "shp_list_ellip" : self.shp_list_ellip, "ellip" : self.if_ellipsoid, "box" : self.if_box}

        plot_set(*pos_params_plot, **name_params_plot, save_fig = save_fig, file_names = file_names)



    def test_box(self, X): ## given a testing data set, check which data samples are within the developed uncertainty set
        coverage_pct, vio_ind = comp_coverage_box(X, self.D_list, self.d_list)

        self.test_pct_box = coverage_pct
        self.test_ind_box = vio_ind 

    def test_ellip(self, X):
        coverage_pct, vio_ind = comp_coverage_ellip(X, self.D_list, self.d_list)

        self.test_pct_ellip = coverage_pct
        self.test_ind_ellip = vio_ind 
        
        

def comput_sqrtM(C_mat): ###  compute the 1/2 inverse of a symmetric matrix
    v, P = np.linalg.eig(C_mat)
    Q = P@np.diag(v**(-0.5))@np.linalg.inv(P)
    return Q