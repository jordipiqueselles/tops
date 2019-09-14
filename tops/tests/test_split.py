from unittest import mock, TestCase
import pytest
import numpy as np
import copy
from sklearn.metrics import accuracy_score

from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_raise_message

from tops.split import SplitPercentile, SplitImpurity, SplitKmeans, SplitMultBinary


class TestSplitPercentile:
    # (idx_var, n_bins, output)
    cases = [(0, 4, [1, 2, 3, 0, 0, 3, 1, 2]),  # numerical variable
             (1, 10, [1, 0, 3, 2, 4, 2, 4, 1]),  # numerical variable, more bins than different values
             (2, 2, [0, 1, 1, 0, 1, 0, 1, 1])]  # binary variable

    @pytest.mark.parametrize('idx_var, n_bins, output', cases)
    def test_split_percentile(self, generate_X_y, idx_var, n_bins, output):
        X, y, metadata = generate_X_y

        # Test the split percentile applied to the `idx_var` column
        split_obj = SplitPercentile(n_bins)
        list_splits = split_obj.get_split(X, y, idx_var, metadata)
        # The split percentile only outputs a single way to split the data
        assert_equal(len(list_splits), 1)
        fun, n_sons = list_splits[0]

        # The number of expected sons is n_bins if there are enough different values, otherwise it's the number
        # of different values
        n_unique_vals = len(np.unique(X[:, 1]))
        assert_equal(n_sons, min(n_bins, n_unique_vals))
        # Check the output is the expected one
        class_X = fun(X)
        assert_array_equal(class_X, output)


class TestSplitMultBinary:
    # (idx_var, l_perc, output)
    cases = [(0, [0.3, 0.6], [[1, 1, 1, 0, 0, 1, 1, 1], [0, 1, 1, 0, 0, 1, 0, 1]]),  # numerical variable
             (1, [0.01], []),  # numerical variable, not enough instances for son_0
             (2, [0.2, 0.4, 0.7], [[0, 1, 1, 0, 1, 0, 1, 1]] * 3)]  # binary variable, in the 3 cases, the same split

    def _get_classification_split(self, split_obj, X, y, idx_var, metadata):
        list_splits = split_obj.get_split(X, y, idx_var, metadata)
        for fun, n_sons in list_splits:
            class_X = fun(X)
            yield class_X

    # Test that SplitMultBinary behaves as SplitPercentile when the unique threshold is 0.5 and n_bins is 2
    @pytest.mark.parametrize('idx_var', [0, 1, 2])
    def test_when_split_mult_binary_equals_split_percentile(self, generate_pseudorandom_X_y, idx_var):
        np.random.seed(0)
        X, y, metadata = generate_pseudorandom_X_y

        split_obj_mult_bin = SplitMultBinary([0.5])
        class_X_mult_bin = list(self._get_classification_split(split_obj_mult_bin, X, y, idx_var, metadata))

        split_obj_perc = SplitPercentile(2)
        class_X_perc = list(self._get_classification_split(split_obj_perc, X, y, idx_var, metadata))

        assert_equal(len(class_X_mult_bin), 1)
        assert_equal(len(class_X_perc), 1)
        assert_array_equal(class_X_mult_bin[0], class_X_perc[0])

    def test_incremental_splits(self, generate_pseudorandom_X_y):
        np.random.seed(0)
        X, y, metadata = generate_pseudorandom_X_y
        l_perc = np.arange(0.0001, 1, 0.007)
        idx_var = 0

        split_obj = SplitMultBinary(l_perc)
        clss_splits = self._get_classification_split(split_obj, X, y, idx_var, metadata)
        tmp_class = np.ones(X.shape[0])
        for clss in clss_splits:
            # Check all existing 0 are preserved in the next iteration
            aux = (tmp_class - clss) >= 0
            assert aux.all()
            tmp_class = clss

    @pytest.mark.parametrize('idx_var, l_perc, outputs', cases)
    def test_split_multi_binary(self, generate_X_y, idx_var, l_perc, outputs):
        X, y, metadata = generate_X_y

        # Test the split percentile applied to the `idx_var` column
        split_obj = SplitMultBinary(l_perc)
        clss_splits = list(self._get_classification_split(split_obj, X, y, idx_var, metadata))

        for clss_split, exp_output in zip(clss_splits, outputs):
            assert_array_equal(clss_split, exp_output)


def test_split_impurity(generate_X_y):
    X, y, metadata = generate_X_y

    # Calculate the minimum impurity with a minimum of one instance per subset
    split_obj = SplitImpurity(1)
    list_splits = split_obj.get_split(X, y, 1, metadata)
    # The split percentile only outputs a single way to split the data
    assert_equal(len(list_splits), 1)
    fun, n_sons = list_splits[0]
    assert_equal(n_sons, 2)
    class_X = fun(X)
    assert_array_equal(class_X, [0, 0, 1, 0, 1, 0, 1, 0])

    # Calculate the minimum impurity with more instances per subset than the total number of instances
    # The result is that no split can be found
    split_obj = SplitImpurity(X.shape[0] // 2 + 1)
    list_splits = split_obj.get_split(X, y, 1, metadata)
    # The split percentile only outputs a single way to split the data
    assert_equal(len(list_splits), 1)
    fun, n_sons = list_splits[0]
    assert_equal(n_sons, 1)


@pytest.mark.skip(reason="Split categorical is not fully implemented yet")
@pytest.mark.parametrize('idx_var', [0, 1, 2])
def test_split_categorical(generate_X_categorical_y, idx_var):
    X, y, metadata = generate_X_categorical_y

    # It doesn't matter the type of split, since the split categorical is the same among them
    split_obj = SplitPercentile(5)
    fun, n_sons = split_obj.get_split(X, y, idx_var, metadata)
    assert_equal(n_sons, 3)
    class_X = fun(X)
    assert_array_equal(class_X, [0, 0, 0, 1, 1, 1, 2, 2])
