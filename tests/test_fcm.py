#!usr/bin/python 3.6
#-*-coding:utf-8-*-

import pytest 
import sys 
import os 
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from size_constrained_clustering import fcm

class TestFCM:

    def test_input(self):
        with pytest.raises(AssertionError):
            fcm.FCM(n_clusters=-1)
        with pytest.raises(AssertionError):
            fcm.FCM(n_clusters=0)
        with pytest.raises(AssertionError):
            fcm.FCM(n_clusters=3, m=1)
        with pytest.raises(AssertionError):
            fcm.FCM(n_clusters=3, max_iters=1.0)
        with pytest.raises(AssertionError):
            fcm.FCM(n_clusters=3, epsilon=-1)
        with pytest.raises(Exception):
            fcm.FCM(n_clusters=3, distance_func="a")
    
    def test_output(self):
        from sklearn.datasets import make_blobs
        import numpy as np 
        import collections 
        n_samples = 5000
        n_bins = 4  # use 3 bins for calibration_curve as we have 3 clusters here
        centers = [(-5, -5), (0, 0), (5, 5), (7, 10)]

        X, _ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                        centers=centers, shuffle=False, random_state=42)

        model = fcm.FCM(n_bins)
        model.fit(X)
        fcm_centers = model.cluster_centers_
        fcm_labels = model.labels_

        target_centers = np.array([[-0.020799, -0.03094044],
                [-4.99797698, -4.96240717],
                [7.01237337, 10.03848252],
                [4.97931177, 4.94258691]])
        # within tolerance
        fcm_centers = np.round(fcm_centers, 3)
        target_centers = np.round(target_centers, 3)
        label_counts = dict(collections.Counter(fcm_labels))
        assert label_counts == {2: 1252, 0: 1250, 1: 1249, 3: 1249}
        assert np.array_equal(fcm_centers, target_centers)

if __name__ == "__main__":
    pass
