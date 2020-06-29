import numpy as np
from sklearn_import.fixes import _parse_version

np_version = _parse_version(np.__version__)


def sparse_min_max(X, axis):
    return (X.min(axis=axis).toarray().ravel(),
            X.max(axis=axis).toarray().ravel())
