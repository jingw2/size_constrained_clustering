#!usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
@file: shrinkage.py, shrinkage clustering
@Author: Jing Wang (jingw2@foxmail.com)
@Date: 06/24/2020
@Paper reference: Shrinkage Clustering: A fast and \
    size-constrained clustering algorithm for biomedical applications
'''

import base 
from scipy.spatial.distance import cdist
import numpy as np
import random

class Shrinkage(base.Base):

    def __init__(self, n_clusters, min_size=1, max_iters=1000, \
        distance_func=cdist, random_state=42):
        '''
        Args:
            n_clusters (int): number of clusters 
            max_iters (int): maximum iterations
            distance_func (object): callable function with input (X, centers) / None, by default is l2-distance
            random_state (int): random state to initiate, by default it is 42
        '''
        super(Shrinkage, self).__init__(n_clusters, max_iters, distance_func)
        np.random.seed(random_state)
        random.seed(random_state)
        self.min_size = min_size

    def fit(self, X):
        
        n_samples, n_features = X.shape
        # calculate similarity matrix, larger similarity means more resemblance
        S = self.distance_func(X, X)
        S /= np.max(S)
        S = 1 - S
        # initialize
        A, S_tilde = self._init(S)
        iters = 0
        while True:
            # remove empty clusters 
            cluster_size = np.sum(A, axis=0)
            keep_cluster = np.where(cluster_size >= self.min_size)[0]
            A = A[:, keep_cluster]
            
            # permute cluster membership
            M = S_tilde @ A
            v = np.min(M - np.sum(M * A, axis=1).reshape((-1, 1)), axis=1)
            X_bar = np.argmin(v)
            C_prime = np.argmin(M[X_bar])

            K = A.shape[1]
            A[X_bar] = np.zeros(K)
            A[X_bar, C_prime] = 1

            if abs(np.sum(v)) < 1e-5  or iters >= self.max_iters:
                break 
            
            iters += 1
        
        self.labels_ = np.argmax(A, axis=1)
        self.cluster_centers_ = self.update_centers(X, A)

    
    def _init(self, S):
        '''
        Initialize A and S_tilde
        '''
        n_samples, _ = S.shape 
        A = np.zeros((n_samples, self.n_clusters))
        A[range(n_samples), [random.choice(range(self.n_clusters)) \
            for _ in range(n_samples)]] = 1 
        S_tilde = 1 - 2 * S 
        return A, S_tilde
    
    def update_centers(self, X, labels):
        '''
        Update centers 
        Args:
            X (array like): (n_samples, n_features)
            labels (array like): (n_samples, n_clusters), one-hot array
        
        Return:
            centers (array like): (n_clusters, n_features)
        '''
        centers = (X.T.dot(labels)).T / np.sum(labels, axis=0).reshape((-1, 1))
        return centers

if __name__ == "__main__":
    from seaborn import scatterplot as scatter
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    n_samples = 1000
    n_clusters = 3

    centers = [(-5, -5), (0, 0), (5, 5), (10, 10)]

    X, _ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                    centers=centers, shuffle=False, random_state=42)

    min_size = 100
    shrink = Shrinkage(n_clusters, min_size)
    shrink.fit(X)

    fcm_centers = shrink.cluster_centers_
    fcm_labels = shrink.labels_

    # plot result
    f, axes = plt.subplots(1, 2, figsize=(11,5))
    scatter(X[:,0], X[:,1], ax=axes[0])
    scatter(X[:,0], X[:,1], ax=axes[1], hue=fcm_labels)
    scatter(fcm_centers[:,0], fcm_centers[:,1], ax=axes[1],marker="s",s=200)
    plt.show()
