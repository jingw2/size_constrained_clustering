#!usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
@file: base.py, base for clustering algorithm
@Author: Jing Wang (jingw2@foxmail.com)
@Date: 06/07/2020
'''
from scipy.spatial.distance import cdist
import numpy as np 
import warnings
import scipy.sparse as sp
from sklearn_import.utils.extmath import stable_cumsum

class Base(object):

    def __init__(self, n_clusters, max_iters, distance_func=cdist):
        '''
        Base Cluster object

        Args:
            n_clusters (int): number of clusters 
            max_iters (int): maximum iterations
            distance_func (callable function): distance function callback
        '''
        assert isinstance(n_clusters, int)
        assert n_clusters >= 1
        assert isinstance(max_iters, int)
        assert max_iters >= 1
        self.n_clusters = n_clusters 
        self.max_iters = max_iters
        if distance_func is not None and not callable(distance_func):
            raise Exception("Distance function is not callable")
        self.distance_func = distance_func

    def fit(self, X):
        pass 

    def predict(self, X):
        pass 

def k_init(X, n_clusters, x_squared_norms, random_state=42, distance_func=cdist, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : integer
        The number of seeds to choose

    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = distance_func(
        centers[0, np.newaxis], X)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        # distance_to_candidates = euclidean_distances(
        #     X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
        distance_to_candidates = distance_func(X[candidate_ids], X)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers
