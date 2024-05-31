import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from sklearn import cluster

from sklearn import preprocessing

import alphashape
from shapely.geometry import Polygon, Point, LineString

from descartes import PolygonPatch
import alphashape
import shapely
from sklearn.mixture import GaussianMixture

import time

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score




class DGP_Set():
    def __init__(self, eps: float, MinPts: int, K: int, if_ellipsoid: bool = False, if_box: bool = True):
        self.X = None
        self.m = None
        self.n = None
        self.eps = eps
        self.MinPts = MinPts
        self.K = K
        self.if_ellipsoid = if_ellipsoid
        self.if_box = if_box
        self.coverage_ellip = None
        self.coverage_box = None
        self.outlier_pct = None
        self.X_list = []
        self.outliers = None
        self.ss = None
        self.D_list = []
        self.d_list = []
        self.beta_list = []
        self.SigmaInv2_list = []
        self.varrho_list = []
        self.shp_list_box = []
        self.shp_list_ellip = []
        

    def fit(self, X, if_save: bool = False):
        self.X = X
        self.m = X.shape[1]
        self.n = X.shape[0]
        dbscan = cluster.DBSCAN(eps = self.eps, min_samples = self.MinPts).fit(self.X)
        X_clean = self.X[dbscan.labels_ != -1, :]
        outliers = self.X[dbscan.labels_ == -1, :]
        self.outliers = outliers

        self.outlier_pct = len(outliers)/self.n * 100

        gmm = GaussianMixture(n_components = self.K, covariance_type = "full", n_init = 100)
        gmm.fit(X_clean)
        cls_labels = gmm.predict(X_clean)
        self.ss = silhouette_score(X_clean, cls_labels)

        for i in range(self.K):
            self.X_list.append(X_clean[cls_labels == i,:])

        if self.if_ellipsoid:
            for k in range(self.K):
                X = self.X_list[k]
                X_mean = gmm.means_[k,:]
                Sigma = gmm.covariances_[k]
                Sigma_inv2 =  comput_sqrtM(Sigma) #np.linalg.inv(Sigma)
                varrho = np.max(np.array([np.linalg.norm(Sigma_inv2@(X[t,:] - X_mean), 2) for t in range(X.shape[0])]))
        
                self.beta_list.append(X_mean)
                self.SigmaInv2_list.append(Sigma_inv2)
                self.varrho_list.append(varrho)

                if self.m == 2:
                    X_centered = X - X_mean
                    pc1_range = np.linspace(np.min(X_centered[:,0])*(1+0.5),np.max(X_centered[:,0])*(1+0.5),500) + X_mean[0]
                    pc2_range = np.linspace(np.min(X_centered[:,1])*(1+0.5),np.max(X_centered[:,1])*(1+0.5),500) + X_mean[1]
                    
                    Points_ellip = []
                    
                    for k in range(len(pc1_range)):
                        for j in range(len(pc2_range)):
                            point = np.array([pc1_range[k], pc2_range[j]])
                            if np.linalg.norm(Sigma_inv2@(point - X_mean), 2) <= varrho:
                                Points_ellip.append(point)
                            
                    shp = alphashape.alphashape(Points_ellip, alpha = 0.)
                    self.shp_list_ellip.append(shp)            
                
            if if_save:
                np.savez("ellipsoid.npz", beta_list = self.beta_list, SigmaInv2_list = self.SigmaInv2_list, varrho_list = self.varrho_list)

            

        
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

            if if_save:
                np.savez("box.npz", D_list = self.D_list, d_list = self.d_list)
        
        
        self.comp_coverage()
        

    
    def plot_set(self, savefig = False):  ####### only support visualization of 2D visualization
        
        assert self.m == 2, "the dimension of the uncertainty is not equal to 2, visualization is not supported"
        
        if self.if_box:

            fig, axs = plt.subplots(figsize=(4,4))
            axs.scatter(self.outliers[:,0], self.outliers[:,1], 0.5, color = "k", label = "outliers")
            
            for k in range(self.K):   
                shp = self.shp_list_box[k]
                X = self.X_list[k]

                axs.scatter(X[:,0], X[:,1], s = 0.5, color = f"C{k}")
                
                if isinstance(shp, Polygon):
                    shp_x, shp_y = shp.exterior.xy
                    axs.fill(shp_x, shp_y, alpha = 0.3, color = f"C{k}", label = "set {k}".format(k = k+1)) 
                    
                if isinstance(shp, LineString):
                    axs.plot(shp.coords, linewidth = 1, label = "set {k}".format(k = k+1))
            
            axs.legend(loc = 2, frameon = False)
            axs.set(
                    aspect="equal",
                    xlabel="first feature",
                    ylabel="second feature",
                    title = "uncertainty set with box")
            
            if savefig:
                plt.savefig("box_dgp_{0}".format(self.MinPts)+".pdf",bbox_inches = "tight")
                

        if self.if_ellipsoid:      ##### plot ellipsoidal subsets
            fig, axs = plt.subplots(figsize = (4,4))
            axs.scatter(self.outliers[:,0], self.outliers[:,1], 0.5, color = "k", label = "outliers")

            for k in range(self.K):
                X = self.X_list[k]
                shp = self.shp_list_ellip[k]
                axs.scatter(X[:,0], X[:,1], 1, color = f"C{k}")
                shp_x, shp_y = shp.exterior.xy
                axs.fill(shp_x, shp_y, alpha = 0.3, color = f"C{k}", label = "set {k}".format(k = k+1)) 
                
                if isinstance(shp, LineString):
                    axs.plot(shp.coords,linewidth = 1, label = "set {k}".format(k = k+1))
            
            
            axs.legend(loc = 2, frameon = False)
    
            
            axs.set(
            aspect="equal",
            #    title="2-dimensional dataset with principal components",
            xlabel="first feature",
            ylabel="second feature",
            title = "uncertainty set with ellipsoid")
               
            plt.savefig("ellip_dpg_"+"{0}".format(self.MinPts)+".pdf",bbox_inches = "tight")
                

    def comp_coverage(self):

        if self.if_ellipsoid:
            inside_num_ellip = 0
            for _, point in enumerate(self.X):
                vio_max = 0
                vio_list = []
    
                for k in range(self.K):

                    _vio = np.linalg.norm(self.SigmaInv2_list[k]@(point - self.beta_list[k]), 2) - self.varrho_list[k] 
                    vio_list.append(_vio)
    
                vio = min(vio_list)
                if vio <= vio_max:
                    inside_num_ellip += 1
            self.coverage_ellip = inside_num_ellip/self.n
            
        if self.if_box:
            inside_num_box = 0
            for _, point in enumerate(self.X):
                vio_max = 1e-2
                vio_list = []
    
                for k in range(self.K):
                    _vio = np.max(self.D_list[k]@point - self.d_list[k])
                    vio_list.append(_vio)
    
                vio = min(vio_list)
                if vio <= vio_max:
                    inside_num_box += 1
    
            self.coverage_box = inside_num_box/(self.n)
        


def comput_sqrtM(C_mat): ###  compute the 1/2 inverse of a symmetric matrix
    v, P = np.linalg.eig(C_mat)
    Q = P@np.diag(v**(-0.5))@np.linalg.inv(P)
    return Q