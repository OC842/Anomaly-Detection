import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

#Creating random sample of data similar to real data
import numpy.random as rnd

size = 100
x = rnd.random(size)

s = rnd.random(size)
t = rnd.random(size)
u = rnd.random(size)

i = 0
while(i < 10):
	h = rnd.randint(0, size, 1)
	x[h] = x[h] + rnd.randint(0, 5, 1)
	i += 1

y = 0
a = []

while(y < size):
	a.append(y)
	y += 1
	
b = []

for i in range(len(s)):
	b.append([s[i], t[i], x[i]])
	
#Transformed into Numpy array for ease
data = np.array(b)





fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
plt.title('Random Data')



#Important line as it compresses the data for a better dbscan fit
data = StandardScaler().fit_transform(data)


# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=1, min_samples=5).fit(data)

#DBSCAN creates 2 arrays. One of all the indices of the points within clusters. The other is 
# an array containing a label for every item.

#This creates an array in the same shape as the label array but every item is false
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

#This changes the false values to true if the point is part of a cluster.
#(and leaves it false if it is an outlier - i.e. does not appear in the index array.
core_samples_mask[db.core_sample_indices_] = True

#Copying the labels array for manipulation
labels = db.labels_



# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

#If the cluster values are already known these scoring functions can be used.

#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels))
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
		  


tilde = []
non_tilde = []	
Outliers = []

for k in unique_labels:
	
	class_member_mask = (labels == k)
	
	if k == -1:
		Outliers.append(data[class_member_mask & ~core_samples_mask])
	else:
	
		non_tilde.append(data[class_member_mask & core_samples_mask])
		
		tilde.append(data[class_member_mask & ~core_samples_mask])
	


total_clusters = len(non_tilde)

dimensions = len(non_tilde[0][0])

lengths = np.ndarray((total_clusters, 2), dtype = int)

Clusters = []

for i in range(total_clusters):

	lengths[i][0] = len(non_tilde[i])
	lengths[i][1] = len(tilde[i])
	
print(lengths)
	

for p in range(total_clusters):

	first = non_tilde[p]
	fisrt = tilde[p]
		
	x = lengths[p,0] + lengths[p][1]
	y = dimensions
		
	c = np.ndarray(shape = (x, y))
		

	
	for i in range(lengths[p][0]):

		for j in range(dimensions):
		
			c[i][j] = first[i][j]
			
			
	for i in range(lengths[p][1]):
		for j in range(dimensions):
			c[i + lengths[p][0]][j] = fisrt[i][j]

		
	Clusters.append(c)

	

fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
for i in range(len(Clusters)):

	ax.scatter(Clusters[i][:,0], Clusters[i][:,1], Clusters[i][:,2], c = [colors[i]])
	ax.scatter(Outliers[0][:,0], Outliers[0][:,1], Outliers[0][:,2], c = ['#000000'])

	
	
plt.title('Number of outliers: %d' % n_noise_)
plt.show()
