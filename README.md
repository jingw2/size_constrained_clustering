## Size Constrained Clustering Solver

Implementation of Size Constrained Clustering. 
Size constrained clustering can be treated as an optimization problem. Details could be found in a set of reference paper.

### Installation
Requirement Python >= 3.6
* Method 1: install from PyPI
  
* Method 2: Download Source Files, and run the following code in terminal
```shell
pip install requirements.txt
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
import size_constrained_clustering
import numpy as np

# initialization
n_points = 1000
X = np.random.rand(n_points, 2)
demands = np.ones((n_points, 1))
n_clusters = 4
n_iters = 100
max_size = [n_points / n_clusters] * n_clusters

da = size_constrained_clustering.DeterministicAnnealing(n_clusters, max_size, n_iters, "l2")
labels, centers = da.fit(X, demands)
```

## Copyright
Copyright (c) 2020 Jing Wang. Released under the MIT License
Third-party copyright in this distribution is noted where applicable.

### Results Show

### Reference
* [Clustering with Capacity and Size Constraints: A Deterministic
Approach](http://web.eecs.umich.edu/~mayankb/docs/ClusterCap.pdf)
* [Deterministic Annealing, Clustering and Optimization](https://thesis.library.caltech.edu/2858/1/Rose_k_1991.pdf)
* [Shrinkage Clustering](https://www.researchgate.net/publication/322668506_Shrinkage_Clustering_A_fast_and_size-constrained_clustering_algorithm_for_biomedical_applications)
* [Clustering with size constraints](https://www.researchgate.net/publication/268292668_Clustering_with_Size_Constraints)
* [Data Clustering with Cluster Size Constraints Using a Modified k-means Algorithm](https://core.ac.uk/download/pdf/61217069.pdf)
* [KMeans Constrained Clustering Inspired by Minimum Cost Flow Problem](https://github.com/joshlk/k-means-constrained)
* [Same Size Kmeans Heuristics Methods](https://elki-project.github.io/tutorial/same-size_k_means)
* [Google's Operations Research tools's
`SimpleMinCostFlow`](https://developers.google.com/optimization/flow/mincostflow)
* [Cluster KMeans Constrained](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf)

### TO DO
* [ ] Size constraint API
* [ ] Examples to show
* [ ] Readme Modification, badges, travis CI
* [ ] Upload PyPI
