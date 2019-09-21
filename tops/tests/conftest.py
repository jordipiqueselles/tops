from tops.misc import Type
import numpy as np
from sklearn.utils.testing import assert_equal
import pytest
from utils.datasets import load_arff


@pytest.fixture
def generate_pseudorandom_X_y():
    n = 100
    X = np.random.rand(n, 3)
    X[:, 2] = np.random.binomial(1, 0.5, n)
    y = np.random.binomial(1, 0.5, n)
    metadata = [{'type': Type.numerical}, {'type': Type.numerical}, {'type': Type.binary}]
    return X, y, metadata


@pytest.fixture
def generate_X_y():
    X = np.array([[2, 1, 0],
                  [5, 0, 1],
                  [8, 7, 1],
                  [1, 6, 0],
                  [0, 9, 1],
                  [9, 6, 0],
                  [3, 9, 1],
                  [7, 1, 1]])
    y = np.array([0, 1, 0, 1, 1, 1, 0, 1])
    metadata = [{'type': Type.numerical}, {'type': Type.numerical}, {'type': Type.binary}]
    assert_equal(X.shape[0], y.shape[0])
    return X, y, metadata


@pytest.fixture
def generate_X_categorical_y():
    X = np.array([[1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 1]])
    y = np.array([0, 1, 0, 1, 1, 1, 0, 1])
    idx_modalities = [0, 1, 2]
    metadata = [{'type': Type.categorical, 'idx_modalities': idx_modalities},
                {'type': Type.categorical, 'idx_modalities': idx_modalities},
                {'type': Type.categorical, 'idx_modalities': idx_modalities}]
    assert_equal(X.shape[0], y.shape[0])
    return X, y, metadata


@pytest.fixture
def load_postoperative_dataset():
    X, y = load_arff('./data/postoperative.patient.data.arff', False)
    return X, y, []
