
import numpy as np 
import collections
import os 
import sys 

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from src import equal, da

n_samples = 10000
n_clusters = 4 
X = np.random.rand(n_samples, 2)
distribution = [0.25] * n_clusters
model = da.DeterministicAnnealing(n_clusters, distribution)
model.fit(X)
