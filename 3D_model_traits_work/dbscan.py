#Packages to cluster our models

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import spectral_clustering

#create paths for our data
data_folder = "./work/"
data_xyz = "ft_all.xyz"
data_ply = "ft_all_aligned.ply"

#for kmeans
x,y,z = np.loadtxt(data_folder+data_xyz,skiprows=1, delimiter=' ', unpack=True)

#DBSCAN
pcd = o3d.io.read_point_cloud(data_folder + data_ply)

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])