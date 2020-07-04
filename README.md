## Size Constrained Clustering Solver
[![Build Status](https://travis-ci.org/jingw2/size_constrained_clustering.svg?branch=master)](https://travis-ci.org/jingw2/size_constrained_clustering)
[![PyPI version](https://badge.fury.io/py/size-constrained-clustering.svg)](https://badge.fury.io/py/size-constrained-clustering)
![GitHub](https://img.shields.io/github/license/jingw2/size_constrained_clustering)

Implementation of Size Constrained Clustering. 
Size constrained clustering can be treated as an optimization problem. Details could be found in a set of reference paper.

### Installation
Requirement Python >= 3.6, Numpy >= 1.13, Cython >= 0.29
* install from PyPI
```shell
pip install size-constrained-clustering
```

### Methods
* Fuzzy C-means Algorithm: Similar to KMeans, but use membership probability, not 0 or 1
* Same Size Contrained KMeans Heuristics: Use Heuristics methods to reach same size clustering
* Same Size Contrained KMeans Inspired by Minimum Cost Flow Problem
* Minimum and Maximum Size Constrained KMeans Inspired by Minimum Cost Flow Problem
* Deterministic Annealling Algorithm: Input target cluster distribution, return correspondent clusters
* Shrinkage Clustering: base algorithm and minimum size constraints

### Usage:
```python
# setup
from size_constrained_clustering import fcm, equal, minmax, shrinkage
# by default it is euclidean distance, but can select others
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
```

Fuzzy C-means 
```python
n_samples = 2000
n_clusters = 4
centers = [(-5, -5), (0, 0), (5, 5), (7, 10)]
X, _ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                    centers=centers, shuffle=False, random_state=42)
model = fcm.FCM(n_clusters)
# use other distance function: e.g. haversine distance
# model = fcm.FCM(n_clusters, distance_func=haversine_distances)
model.fit(X)
centers = model.cluster_centers_
labels = model.labels_
```
![alt text](https://github.com/jingw2/size_constrained_clustering/blob/master/pic/fcm.png)


Equal Size Constraint
```python
n_samples = 2000
n_clusters = 3
X = np.random.rand(n_samples, 2)
# use minimum cost flow framework to solve
model = equal.SameSizeKMeansMinCostFlow(n_clusters)
# use heuristics method to solve
model = equal.SameSizeKMeansHeuristics(n_clusters)
model.fit(X)
centers = model.cluster_centers_
labels = model.labels_
```
![alt text](https://github.com/jingw2/size_constrained_clustering/blob/master/pic/equal.png)

Cluster size: 667, 667 and 666 in the figure above.

Minimum and Maximum Size Constraint
```python
n_samples = 2000
n_clusters = 3
X = np.random.rand(n_samples, 2)
model = minmax.MinMaxKMeansMinCostFlow(n_clusters, size_min=400,   size_max=800)
model.fit(X)
centers = model.cluster_centers_
labels = model.labels_
```
![alt text](https://github.com/jingw2/size_constrained_clustering/blob/master/pic/minmax.png)

Cluster size: 753, 645 and 602 in the figure above.

Deterministic Annealing
```python
n_samples = 2000
n_clusters = 3
X = np.random.rand(n_samples, 2)
# distribution is the distribution of cluster sizes
model = da.DeterministicAnnealing(n_clusters, distribution=[0.1, 0.6, 0.3])
model.fit(X)
centers = model.cluster_centers_
labels = model.labels_
```
![alt text](https://github.com/jingw2/size_constrained_clustering/blob/master/pic/da.png)

Cluster size: 1200, 600 and 200 in the figure above, corresponding to distribution [0.6, 0.3, 0.1]

Shrinkage Clustering

The result might be not available.
```python
n_samples = 1000
n_clusters = 4
centers = [(-5, -5), (0, 0), (5, 5), (7, 10)]
X, _ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0, centers=centers, shuffle=False, random_state=42)

model = shrinkage.Shrinkage(n_clusters, size_min=100)
model.fit(X)
centers = model.cluster_centers_
labels = model.labels_
```
![alt text](https://github.com/jingw2/size_constrained_clustering/blob/master/pic/shrinkage.png)


## Copyright
Copyright (c) 2020 Jing Wang. Released under the MIT License. 

Third-party copyright in this distribution is noted where applicable.

### Reference
* [Clustering with Capacity and Size Constraints: A Deterministic
Approach](http://web.eecs.umich.edu/~mayankb/docs/ClusterCap.pdf)
* [Deterministic Annealing, Clustering and Optimization](https://thesis.library.caltech.edu/2858/1/Rose_k_1991.pdf)
* [Deterministic Annealing, Constrained Clustering, and Opthiieation](https://authors.library.caltech.edu/78353/1/00170767.pdf)
* [Shrinkage Clustering](https://www.researchgate.net/publication/322668506_Shrinkage_Clustering_A_fast_and_size-constrained_clustering_algorithm_for_biomedical_applications)
* [Clustering with size constraints](https://www.researchgate.net/publication/268292668_Clustering_with_Size_Constraints)
* [Data Clustering with Cluster Size Constraints Using a Modified k-means Algorithm](https://core.ac.uk/download/pdf/61217069.pdf)
* [KMeans Constrained Clustering Inspired by Minimum Cost Flow Problem](https://github.com/joshlk/k-means-constrained)
* [Same Size Kmeans Heuristics Methods](https://elki-project.github.io/tutorial/same-size_k_means)
* [Google's Operations Research tools's
`SimpleMinCostFlow`](https://developers.google.com/optimization/flow/mincostflow)
* [Cluster KMeans Constrained](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf)
