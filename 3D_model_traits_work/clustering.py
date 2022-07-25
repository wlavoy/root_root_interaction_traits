'''
First attempts to cluster two root systems to use in Suxing's model traits pipeline.
Here my goal is to separate the two root systems used in my root-root interaction studies.
By separating the roots the pipeline can quantify traits of interest separately and with respects
to one another.

Here I am using KMeans
--William LaVoy--
'''



# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

#create paths for our data
data_folder = "./work/"
data = "ft_all.xyz"

x,y,z = np.loadtxt(data_folder+data,skiprows=1, delimiter=' ', unpack=True)

'''
#2d plots to look at view
plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.scatter(x, z, s=0.05)
plt.axhline(y=np.mean(z), color='r', linestyle='-')
plt.title("First view")
plt.xlabel('X-axis')
plt.ylabel('Z-axis')

plt.subplot(1, 2, 2) # index 2
plt.scatter(y, z, s=0.05)
plt.axhline(y=np.mean(z), color='r', linestyle='-')
plt.title("Second view")
plt.xlabel('Y-axis')
plt.ylabel('Z-axis')

plt.show()




#plotting the results 3D
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, s=0.1)
plt.show()


#DBSCAN implementation
X=np.column_stack((x, y, z))

clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=clustering.labels_, s=20)
plt.show()

'''

#Kmeans implementation
X=np.column_stack((x, y, z))

kmeans = KMeans(n_clusters=2).fit(X)
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=kmeans.labels_, s=0.1)
plt.show()

#catches a small segment of the other root system
'''
#DBSCAN implementation
X=np.column_stack((x, y))
clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
plt.scatter(x, y, c=clustering.labels_, s=20)
plt.show()
'''