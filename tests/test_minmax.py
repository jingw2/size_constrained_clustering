#!usr/bin/python 3.6
#-*-coding:utf-8-*-

import pytest 
import sys 
import os 
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from src import minmax

class TestMinMax:

    def test_input(self):
        with pytest.raises(AssertionError):
            minmax.MinMaxKMeansMinCostFlow(n_clusters=-1, size_min=1, size_max=2)
        with pytest.raises(AssertionError):
            minmax.MinMaxKMeansMinCostFlow(n_clusters=0, size_min=1, size_max=2)
        with pytest.raises(AssertionError):
            minmax.MinMaxKMeansMinCostFlow(n_clusters=1, max_iters=1.2, size_min=1, size_max=2)
        with pytest.raises(Exception):
            minmax.MinMaxKMeansMinCostFlow(n_clusters=1, distance_func="a", size_min=1, size_max=2)
        with pytest.raises(AssertionError):
            minmax.MinMaxKMeansMinCostFlow(n_clusters=1, size_min=None, size_max=2)
        with pytest.raises(AssertionError):
            minmax.MinMaxKMeansMinCostFlow(n_clusters=1, size_min=-1, size_max=2)
        with pytest.raises(AssertionError):
            minmax.MinMaxKMeansMinCostFlow(n_clusters=1, size_min=20, size_max=10)
        with pytest.raises(AssertionError):
            model = minmax.MinMaxKMeansMinCostFlow(n_clusters=1, size_min=10, size_max=20)
            import numpy as np 
            X = np.random.random((1000, 2))
            model.fit(X)
    
    def test_output(self):
        from sklearn.datasets import make_blobs
        import collections 

        n_samples = 2000
        n_clusters = 4  # use 3 bins for calibration_curve as we have 3 clusters here
        centers = [(-5, -5), (0, 0), (5, 5), (7, 10)]

        X, _ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                        centers=centers, shuffle=False, random_state=42)
        
        minsize = 200
        maxsize = 800
        model = minmax.MinMaxKMeansMinCostFlow(n_clusters, size_min=minsize, 
            size_max=maxsize)
        model.fit(X)

        label_counter = collections.Counter(model.labels_)

        for label, count in label_counter.items():
            assert count >= minsize and count <= maxsize
