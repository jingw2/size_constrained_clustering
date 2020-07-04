#!usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
@file: shrinkage.py, shrinkage clustering
@Author: Jing Wang (jingw2@foxmail.com)
@Date: 06/24/2020
@Paper reference: Shrinkage Clustering: A fast and \
    size-constrained clustering algorithm for biomedical applications
'''

import os 
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
import base 
from scipy.spatial.distance import cdist
import numpy as np
import random

class Shrinkage(base.Base):

    def __init__(self, n_clusters, size_min=1, max_iters=1000, \
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
        self.size_min = size_min
        assert isinstance(size_min, int)
        assert size_min >= 1 

    def fit(self, X):
        
        n_samples, n_features = X.shape

        assert self.size_min <= n_samples // self.n_clusters
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
            keep_cluster = np.where(cluster_size >= self.size_min)[0]
            A = A[:, keep_cluster]
            
            # permute cluster membership
            M = S_tilde @ A
            v = np.min(M - np.sum(M * A, axis=1).reshape((-1, 1)), axis=1)
            X_bar = np.argmin(v)
            C_prime = np.argmin(M[X_bar])

            K = A.shape[1]
            A[X_bar] = np.zeros(K)
            A[X_bar, C_prime] = 1

            if abs(np.sum(v)) < 1e-5 or iters >= self.max_iters:
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
        A[range(n_samples), [random.choice(range(self.n_clusters)) for _ in range(n_samples)]] = 1
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
