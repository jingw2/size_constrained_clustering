#!usr/bin/python 3.6
#-*-coding:utf-8-*-

import pytest 
import sys 
import os 
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from src import equal

class TestEqual:

    def test_input(self):
        with pytest.raises(AssertionError):
            equal.SameSizeKMeansHeuristics(n_clusters=-1)
        with pytest.raises(AssertionError):
            equal.SameSizeKMeansMinCostFlow(n_clusters=-1)
        with pytest.raises(AssertionError):
            equal.SameSizeKMeansHeuristics(n_clusters=0)
        with pytest.raises(AssertionError):
            equal.SameSizeKMeansMinCostFlow(n_clusters=0)
        with pytest.raises(AssertionError):
            equal.SameSizeKMeansHeuristics(n_clusters=1, max_iters=1.2)
        with pytest.raises(AssertionError):
            equal.SameSizeKMeansMinCostFlow(n_clusters=1, max_iters=1.2)
        with pytest.raises(Exception):
            equal.SameSizeKMeansHeuristics(n_clusters=1, distance_func="a")
        with pytest.raises(Exception):
            equal.SameSizeKMeansMinCostFlow(n_clusters=1, distance_func="a")
    
    def test_output(self):
        import numpy as np 
        import collections
        n_samples = 2000
        n_clusters = 4 
        X = np.random.rand(n_samples, 2)
        model = equal.SameSizeKMeansHeuristics(n_clusters)
        model.fit(X)
        labels = model.labels_
        label_counts = collections.Counter(labels)
        assert_cluster_equal(label_counts)

        model = equal.SameSizeKMeansMinCostFlow(n_clusters)
        model.fit(X)
        labels = model.labels_
        label_counts = collections.Counter(labels)
        assert_cluster_equal(label_counts)

def assert_cluster_equal(label_counts):
    size = label_counts[0]
    for i in range(1, len(label_counts)):
        assert label_counts[i] == size
        
