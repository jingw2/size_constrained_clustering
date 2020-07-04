#!usr/bin/python 3.7
#-*-coding:utf-8-*-

import pytest 
import sys 
import os 
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from size_constrained_clustering import da

class TestDA:

    def test_input(self):
        with pytest.raises(AssertionError):
            da.DeterministicAnnealing(n_clusters=2, distribution=[0.25, 0.3])
        with pytest.raises(AssertionError):
            da.DeterministicAnnealing(n_clusters=1, distribution=[0.25, 0.3])
        with pytest.raises(AssertionError):
            da.DeterministicAnnealing(n_clusters=2, distribution=[0.25, 0.75], T=0.1)
    
    def test_output(self):
        import collections
        import random 
        import numpy as np 
        n_samples = 1000
        random_state = 42
        random.seed(random_state)
        np.random.seed(random_state)
        X = np.random.rand(n_samples, 2)
        n_clusters = 4
        distribution = [0.25] * n_clusters

        model = da.DeterministicAnnealing(n_clusters, distribution)
        model.fit(X)

        labels = model.labels_
        label_counter = collections.Counter(labels)
        label_dist = list(label_counter.values())
        label_dist = [d / np.sum(label_dist) for d in label_dist]

        assert np.sum(np.array(label_dist) - np.array(distribution)) <= 1e-6
