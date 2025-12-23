########################################################################
###  plot the uncertainty set 
########################################################################
import matplotlib.pyplot as plt

import alphashape
from shapely.geometry import Polygon, Point, LineString

from descartes import PolygonPatch
import alphashape
import shapely


def plot_set(m, outliers, X_list, shp_list_box = None, shp_list_ellip = None, ellip = False, box = True, save_fig = False, file_names = ["box_dgp.pdf","ellip_dpg.pdf"]):  ####### only support visualization of 2D visualization

    if m != 2:
        print("\n The provided data samples are not supported. Only 2-D samples are supported for visualization")
        
    K = len(X_list)
    if box:    ### plot box uncertainty subsets
        fig, axs = plt.subplots(figsize=(5,5))
        axs.scatter(outliers[:,0], outliers[:,1], 0.5, color = "k", label = "outliers")
        
        for k in range(K):   
            shp = shp_list_box[k]
            X = X_list[k]

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

        plt.show()
            
        if save_fig:
            plt.savefig(file_names[0],bbox_inches = "tight")

                

    if ellip:      ### plot ellipsoidal uncertainty subsets
        fig, axs = plt.subplots(figsize = (5,5))
        axs.scatter(outliers[:,0], outliers[:,1], 0.5, color = "k", label = "outliers")

        for k in range(K):
            X = X_list[k]
            shp = shp_list_ellip[k]
            axs.scatter(X[:,0], X[:,1], 1, color = f"C{k}")
            
            shp_x, shp_y = shp.exterior.xy
            axs.fill(shp_x, shp_y, alpha = 0.3, color = f"C{k}", label = "set {k}".format(k = k+1)) 
            
            if isinstance(shp, LineString):
                axs.plot(shp.coords,linewidth = 1, label = "set {k}".format(k = k+1))
        
        
        axs.legend(loc = 2, frameon = False)

        
        axs.set(
            aspect="equal",
            xlabel="first feature",
            ylabel="second feature",
            title = "uncertainty set with ellipsoid")
        plt.show()
        if save_fig:   
            plt.savefig(file_names[1],bbox_inches = "tight")