
from size_constrained_clustering import fcm, equal, minmax, shrinkage
import numpy as np
n_samples = 2000
n_clusters = 3
X = np.random.rand(n_samples, 2)
# 使用minmax flow方式求解
model = equal.SameSizeKMeansMinCostFlow(n_clusters)
# 使用heuristics方法求解
model.fit(X)
centers = model.cluster_centers_
labels = model.labels_
