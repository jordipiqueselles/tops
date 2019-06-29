# coding=utf-8
"""
This file contains the classes used to implement the ToPs predictive algorithm.
The main class to use for the users is ToPs
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
from functools import partial
import copy
from scipy.optimize import minimize, Bounds
import pandas as pd
from utils.timing import TimeCounterCollection
import warnings

###############
#### UTILS ####
###############

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def is_numeric(elem):
    """
    Checks whether the given elemen is a numeric, by trying to sum 1
    """
    try:
        x = elem + 1
    except TypeError:
        return False
    return True


############################
#### THE TOPS ALGORITHM ####
############################

class ToPs(BaseEstimator, ClassifierMixin):
    """
    ToPs classifier

    This classifiers builds a tree of predictors. This is a tree where each node has associated a predictor.
    The overall prediction for an instance is given by the aggregation of all the predictors we find in the
    path along the tree, from the root to a leaf, described by this instance.

    Parameters
    ----------
    predictors : list
        A list of base predictors. The algorithm will choose the best one for a given node according to a
        goodness metric.

    split_type : Derived class of tops.BaseSplit
        Policy to use when creating a tentative split over a given variable.

    goodness_metric : sklearn.metrics or compatible
        A function that given an array of y_true and an array of y_pred gives a number that measures the goodness
        of this prediction.

    min_inst : int
        Minimum number of instances in a node

    min_prob_hoeffding : float, default: 0.8
        A probability value, between 0 and 1. A split will be accepted if the probability that the sons outperform
        the parent node in terms of the goodness_metric is greater than min_prob_hoeffding

    cv1 : int or float, default: 0.15
        It specifies the type of validation used to select the best split-predictor.
        If the value is between 0 and 1, cv1 represents the proportion of instances used as validation set 1
        If the value is an integer greater than 1, it represents the number of folds in the CV1

    cv2 : float, default: 0.10
        It represents the proportion of instances for the validation set 2, used to aggregate the predictors

    min_inst_val : int, default: 20
        Minimum number of instances devoted to validation (V1 or V2) in a node

    normalize_data : bool, default: False
        Whether to normalize or not the numerical features in the dataset before the training process

    file_log : str, default: None
        File where to write the logs

    var_pca : float, default: None
        Not implemented
    """

    binary = "binary"
    categorical = "categorical"
    numerical = "numerical"

    ########################
    #### Public methods ####
    ########################

    def __init__(self, predictors, split_type, goodness_metric, min_inst, min_prob_hoeffding=0.8, cv1=0.15, cv2=0.1,
                 min_inst_val=20, normalize_data=False, file_log=None, var_pca=None):
        # Attributes of the class that take values from the function parameters (explained in the class description)
        self.predictors = predictors
        self.split_type = split_type
        self.goodness_metric = goodness_metric
        self.min_inst = min_inst
        self.min_prob_hoeffding = min_prob_hoeffding
        self.cv1 = cv1
        self.cv2 = cv2
        self.min_inst_val = min_inst_val
        self.normalize_data = normalize_data
        self.file_log = file_log
        self.var_pca = var_pca

        # We need to wrap the predictors to deal with cases when we have only one class, for example
        self.predictors_wrapped = [WrapperPredictor(predictor) for predictor in self.predictors]
        # Root node of the tree of predictors
        self.root_node = None
        # Number of different classes in y
        self.n_classes = None
        # List of unique values of y
        self.classes_ = []
        self.enc_classes = []
        # A list where the ith item contains the type of the variable in the ith column of X,
        # before the preprocessing step. The types can be binary, categorical or numerical
        self.orig_col_types = []
        # A list where the ith item contains a dictionary with metainformation about the ith column of X,
        # after the preprocessing step. Keys: 'type', (optional) 'idx_modalities'
        self.metadata = []
        # For the pca decompostion
        self.pca = None

        # Object use to normalize the numerical variables of a dataset
        self.scaler = StandardScaler()
        # Object use to encode as a sequence of numbers, from 0 to n_classes, the classes represented in y
        self.label_encoder = LabelEncoder()
        # Object to encode the categorical variables of a dataset using the one-hot encoding strategy
        self.one_hot_encoder = OneHotEncoder(sparse=False)

        self.timers = TimeCounterCollection(['fit_node', 'fit_base', 'predict_base', 'train_val', 'split_node',
                                             'x_y_son', 'idx_val_son', 'aggregate_node', 'predict_node',
                                             'pre_fit', 'get_split', 'init_fit_node', 'eval_goodness1'])

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. It can contain both numerical
            and categorical features

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
        """
        self.timers.reset_all()
        self.timers.start('pre_fit')
        X, y = shuffle(X, y)

        df_X = pd.DataFrame(X)

        # Analysis of the data, to populate the metadata fields
        self._analyze_data(df_X, y)
        # Preprocess the data to have a correct type and shape
        X = self._preprocess_X(df_X)
        y = self._preprocess_y(y)
        self._learn_transformation(X)
        X = self._transform(X)
        # Check the consistency of the data
        self._check_data_fit(X, y)

        list_val_idx1, list_val_idx2 = self._create_list_val_idx(self.cv1, self.cv2, X.shape[0])
        self.timers.stop('pre_fit')

        self.root_node = Node(X, y, self, list_val_idx1, list_val_idx2, [])
        self.root_node.fit()
        self.root_node.split()
        self.root_node.aggregate([])

        # Return the classifier
        self.root_node.clear_data()
        self.write_log()
        return self

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        # Check is fit had been called
        if self.root_node is None:
            raise AttributeError("The predictor is not fitted")

        # Preprocess the data
        df_X = pd.DataFrame(X)
        X = self._preprocess_X(df_X)
        X = self._transform(X)
        # Input validation
        self._check_data_predict(X)

        prob = self.root_node.predict_proba(X, [])
        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        indices = prob.argmax(axis=1)
        return self.classes_[indices]


    #########################
    #### Private methods ####
    #########################

    def _analyze_data(self, X, y):
        """Analyzes the input data X, y and populates the label_encoder, scaler, classes_ and metadata attributes

        :param X: array-like, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples,)
        """
        self.label_encoder.fit(y)
        self.classes_ = np.unique(y)

        # Analyze the type of values in the columns of X
        self.orig_col_types = []
        for col in X.columns:
            if is_numeric(X[col][0]):
                un_values = np.unique(X.loc[:, col])
                if len(un_values) == 2 and un_values[0] == 0 and un_values[1] == 1:
                    self.orig_col_types.append(self.binary)
                else:
                    self.orig_col_types.append(self.numerical)
            else:
                self.orig_col_types.append(self.categorical)

        # Fit the transformer for the numerical variables
        idx_numerical = [i for i, col_type in enumerate(self.orig_col_types) if col_type == ToPs.numerical]
        if len(idx_numerical) > 0:
            self.scaler.fit(X.iloc[:, idx_numerical])

        # No need for any transformer for the binary ones
        idx_binary = [i for i, col_type in enumerate(self.orig_col_types) if col_type == ToPs.binary]

        # Fit the transformer for the categorical variables
        idx_categorical = [i for i, col_type in enumerate(self.orig_col_types) if col_type == ToPs.categorical]
        if len(idx_categorical) > 0:
            self.one_hot_encoder.fit(X.iloc[:, idx_categorical])

        # Populate metadata
        self.metadata = []
        for _ in range(len(idx_numerical)):
            self.metadata.append({'type': ToPs.numerical})
        for _ in range(len(idx_binary)):
            self.metadata.append({'type': ToPs.binary})
        for i in range(len(idx_categorical)):
            idx_first_mod = len(self.metadata)
            idx_modalities = [idx_first_mod + j for j in range(len(self.one_hot_encoder.categories_[i]))]
            for _ in idx_modalities:
                self.metadata.append({'type': ToPs.categorical, 'idx_modalities': idx_modalities})

    def _preprocess_X(self, X):
        """Preprocess the X matrix based on the information stored in scaler
        """
        # The numerical variables
        idx_numerical = [i for i, col_type in enumerate(self.orig_col_types) if col_type == ToPs.numerical]
        if len(idx_numerical) > 0:
            if self.normalize_data:
                X_num = self.scaler.transform(X.iloc[:, idx_numerical])
            else:
                X_num = X.iloc[:, idx_numerical]
        else:
            X_num = X.iloc[:, []]

        # The binary variables
        idx_binary = [i for i, col_type in enumerate(self.orig_col_types) if col_type == ToPs.binary]
        X_bin = X.iloc[:, idx_binary]

        # The categorical variables
        idx_categorical = [i for i, col_type in enumerate(self.orig_col_types) if col_type == ToPs.categorical]
        if len(idx_categorical) > 0:
            X_cat = self.one_hot_encoder.transform(X.iloc[:, idx_categorical])
        else:
            X_cat = X.iloc[:, []]

        X_preprocessed = np.concatenate((X_num, X_bin, X_cat), axis=1)
        try:
            return X_preprocessed.astype(np.float32)
        except:
            warnings.warn('Could not convert to float32')
            return X_preprocessed

    def _preprocess_y(self, y):
        """Encodes the target variable y using the label_encoder
        """
        y_enc = self.label_encoder.transform(y)
        return y_enc

    def _learn_transformation(self, X):
        '''
        Learns how to apply a transformation over the dataset. Now, the only transformation available is PCA
        :param X: array-like, shape (n_samples, n_features)
        '''
        if self.var_pca is not None:
            self.pca = PCA(self.var_pca)
            # TODO Take only a reasonable amount of instances for fitting the PCA
            self.pca.fit(X)
            self.metadata = [{'type': ToPs.numerical} for _ in range(len(self.pca.components_))]

    def _transform(self, X):
        '''
        Learns how to apply a transformation over the dataset. Now, the only transformation available is PCA
        :param X: array-like, shape (n_samples, n_features)
        '''
        if self.pca is not None:
            X = self.pca.transform(X)
        return X

    def _check_data_fit(self, X, y):
        """Check after the preprocessing step
        """
        # Check that X and y have correct shape
        check_X_y(X, y)

        # Check both X and y are pure np.array (not pd.DataFrame)
        check_array(X)

        self.enc_classes = unique_labels(y)
        self.n_classes = len(self.enc_classes)

        # The classes of y go from 0 to n_classes
        if not all(a == b for a, b in zip(self.enc_classes, range(self.n_classes))):
            raise ValueError("The classes of y must go from 0 to self.n_classes")

    def _check_data_predict(self, X):
        """Check after the preprocessing step
        """
        # Check both X and y are pure np.array (not pd.DataFrame)
        check_array(X)

        # X has the expected number of columns
        if X.shape[1] != len(self.metadata):
            raise ValueError("X doesn't have the expected number of columns")

    @staticmethod
    def _create_list_val_idx(cv1, cv2, n_inst):
        def create_train_val_bitmap(n_inst, offset, idx_start_val, idx_end_val):
            train_bitmap = np.ones(n_inst, dtype=np.bool)
            train_bitmap[:offset] = False
            train_bitmap[offset + idx_start_val:offset + idx_end_val] = False
            val_bitmap = np.zeros(n_inst, dtype=np.bool)
            val_bitmap[offset + idx_start_val:offset + idx_end_val] = True
            return train_bitmap, val_bitmap

        if cv2 < 0.5:
            offset_cv1 = int(n_inst * cv2)
            tuple_val = create_train_val_bitmap(n_inst, 0, 0, offset_cv1)
            list_val_idx2 = [tuple_val]
        elif cv2 >= 2 and int(cv2) == cv2:
            offset_cv1 = 0
            size_fold = int(np.ceil(n_inst / cv2))
            list_val_idx2 = []
            for i in range(cv2):
                tuple_val = create_train_val_bitmap(n_inst, 0, size_fold * i, size_fold * (i + 1))
                list_val_idx2.append(tuple_val)
        else:
            raise ValueError('Invalid value for the cv2 ({})'.format(cv2))

        if cv1 < 0.5:
            n_val_inst = int(n_inst * cv1)
            tuple_val = create_train_val_bitmap(n_inst, offset_cv1, 0, n_val_inst)
            list_val_idx1 = [tuple_val]
        elif cv1 >= 2 and int(cv1) == cv1:
            size_fold = int(np.ceil((n_inst - offset_cv1) / cv1))
            list_val_idx1 = []
            for i in range(cv1):
                tuple_val = create_train_val_bitmap(n_inst, offset_cv1, size_fold * i, size_fold * (i + 1))
                list_val_idx1.append(tuple_val)
        else:
            raise ValueError('Invalid value for the cv1 ({})'.format(cv1))

        return list_val_idx1, list_val_idx2

    def write_log(self):
        if self.file_log is not None:
            with open(self.file_log, 'a') as f:
                f.write(self.__str__())
                f.write('..............................................................\n')

    def __str__(self):
        return self.root_node.__str__()

############################
####### SPLIT TYPES ########
############################

class BaseSplit(ABC):
    """
    Base class for doing a split
    """
    def __repr__(self):
        name = str(self.__class__).split('.')[-1][:-2]
        list_params = []
        for key, value in self.__dict__.items():
            list_params.append(str(key) + '=' + str(value))
        return name + '(' + ', '.join(list_params) + ')'

    def get_split(self, X, y, idx_var, metadata):
        """From the input data X, y it generates a list of splits using the variable idx_var. Each split is represented
        by a function that returns in which bin a new instance must go.
        :param X: array-like, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples,)
        :param idx_var: Index of the variable used to generate the split
        :param metadata: Metadata of the dataset
        :return: A list -> [(decision_fun, n_bins), ...]
        """
        if metadata[idx_var]['type'] == ToPs.binary:
            return self._get_split_binary(X, idx_var)
        elif metadata[idx_var]['type'] == ToPs.categorical:
            return self._get_split_categorical(metadata, idx_var)
        elif metadata[idx_var]['type'] == ToPs.numerical:
            return self._get_split_numerical(X, y, idx_var, metadata)
        else:
            raise NotImplementedError("There's no split for the type of variable " + metadata[idx_var])

    def _get_split_binary(self, X, idx_var):
        n_bins = len(np.unique(X[:, idx_var]))
        decision_fun = self.DecisionFunction(self.decision_binary, idx_var)
        return [(decision_fun, n_bins)]

    def _get_split_categorical(self, metadata, idx_var):
        raise NotImplementedError('_get_split_categorical is not implemented')
        idx_modalities = metadata[idx_var]['idx_modalities']
        return [(partial(self.decision_categorical, idx_modalities=idx_modalities), len(idx_modalities))]

    @abstractmethod
    def _get_split_numerical(self, X, y, idx_var, metadata):
        raise NotImplementedError()

    @staticmethod
    def decision_binary(X, idx_var):
        return X[:, idx_var]

    @staticmethod
    def decision_categorical(X, idx_modalities):
        # An instance of the first modality will be associated with a 0. For the second one, a 1...
        return np.dot(X[:, idx_modalities], np.arange(0, len(idx_modalities), 1))

    @staticmethod
    def decision_numerical(X, idx_var, thresholds):
        return np.digitize(X[:, idx_var], thresholds, right=False)

    class DecisionFunction:
        def __init__(self, fun, idx_var):
            self.fun = fun
            self.idx_var = idx_var

        def __call__(self, X):
            return self.fun(X=X, idx_var=self.idx_var)

class SplitImpurity(BaseSplit):
    """
    Class used to generate a splits base on the reduction of the gini impurity or the entropy
    """
    def __init__(self, imp_fun, min_inst):
        self.imp_fun = imp_fun
        if min_inst < 1:
            raise ValueError("min_inst must be equal or greater than 1")
        self.min_inst = min_inst

    def _get_split_numerical(self, X, y, idx_var, metadata):
        if X.shape[0] < self.min_inst * 2 + 1:
            decision_fun = self.DecisionFunction(partial(self.decision_numerical, thresholds=[np.inf]), idx_var)
            return [(decision_fun, 1)]

        # [(X_col, y), (X_col, y), ...] sorted by X_col
        data = sorted(zip(X[:, idx_var].tolist(), y.tolist()))
        n_labels = y.max() + 1

        # Initialize the class distribution for the two subsets in which the data will be split
        clss_dstr_1 = [0] * n_labels
        for col_value, clss in data[:self.min_inst]:
            clss_dstr_1[clss] += 1
        clss_dstr_2 = [0] * n_labels
        for col_value, clss in data[self.min_inst:]:
            clss_dstr_2[clss] += 1

        last_col_value = data[self.min_inst - 1][0]
        best_gini = np.inf
        best_col_value = np.inf

        for col_value, clss in data[self.min_inst:len(data)-self.min_inst+1]:
            if last_col_value != col_value:
                new_gini = self.gini(clss_dstr_1) + self.gini(clss_dstr_2)
                (best_gini, best_col_value) = min((best_gini, best_col_value), (new_gini, col_value))
            clss_dstr_1[clss] += 1
            clss_dstr_2[clss] -= 1

        decision_fun = self.DecisionFunction(partial(self.decision_numerical, thresholds=[best_col_value]), idx_var)
        if best_col_value == np.inf:
            return [(decision_fun, 1)]
        else:
            return [(decision_fun, 2)]

    @staticmethod
    def gini(clss_distr):
        N = sum(clss_distr)
        return sum(n*(1-n/N) for n in clss_distr)

    @staticmethod
    def entropy():
        raise NotImplementedError()


class SplitPercentile(BaseSplit):
    """
    Class used to generate a split with n equally sized bins
    """
    def __init__(self, n_bins):
        self.n_bins = n_bins
        if self.n_bins < 2:
            raise ValueError("The variable n_bins must be greater or equal than 1")

    def _get_split_numerical(self, X, y, idx_var, metadata):
        col_unique = np.unique(X[:, idx_var])
        # Check if we have enough elements to fill all the bins
        if len(col_unique) > self.n_bins:
            n_bins = self.n_bins
        else:
            n_bins = len(col_unique)

        # Only one distinct value in the selected column. All the data in the same bin
        if n_bins == 1:
            decision_fun = self.DecisionFunction(partial(self.decision_numerical, thresholds=[np.inf]), idx_var)
            return [(decision_fun, 1)]

        # there's a more efficient method that doesn't need to sort the vector
        sort_col = np.sort(col_unique)

        idx_thresholds = np.array([i * len(sort_col) // n_bins for i in range(1, n_bins)])
        thresholds = sort_col[idx_thresholds]

        decision_fun = self.DecisionFunction(partial(self.decision_numerical, thresholds=thresholds), idx_var)
        return [(decision_fun, n_bins)]


class SplitOriginal(BaseSplit):
    """
    This class is used to create the split used in the original description of ToPs.
    It creates 9 binary splits by generating 9 thresholds that divide the dataset into 10 subsets of almost equal size
    """
    def __init__(self, list_percentile):
        self.list_percentile = list_percentile
        if any(percentile <= 0 or 1 <= percentile for percentile in list_percentile):
            raise ValueError("The variable percentile must be between 0 and 1")

    def _get_split_numerical(self, X, y, idx_var, metadata):
        col_unique = np.unique(X[:, idx_var])

        # Only one distinct value in the selected column. All the data in the same bin
        if len(col_unique) == 1:
            decision_fun = self.DecisionFunction(partial(self.decision_numerical, thresholds=[np.inf]), idx_var)
            return [(decision_fun, 1)]

        # there's a more efficient method that doesn't need to sort the vector
        sort_col = np.sort(X[:, idx_var])

        # we take only unique values, to avoid unnecessary duplicated splits
        list_thresholds = sorted(set(sort_col[int(percentile * len(sort_col))] for percentile in self.list_percentile))
        # We remove the first item if it has the value of the first element in the current column in X,
        # because it will generate a split with an empty subset
        if list_thresholds[0] <= sort_col[0]:
            list_thresholds.pop(0)

        list_splits = []
        for threshold in list_thresholds:
            decision_fun = self.DecisionFunction(partial(self.decision_numerical, thresholds=[threshold]), idx_var)
            list_splits.append((decision_fun, 2))
        return list_splits


class SplitKmeans(BaseSplit):
    """
    Class used to create a split using k-means
    """
    def __init__(self):
        raise NotImplementedError()

    def _get_split_numerical(self, X, y, idx_var, metadata):
        raise NotImplementedError()


#####################
####### NODE ########
#####################

class Node:
    """
    It represents an internal node of ToPs

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features. It can contain both numerical
        and categorical features

    y : array-like, shape (n_samples,)
        Target vector relative to X.

    list_val_idx1 : [(array_bool_train, array_bool_val), ...]
        Arrays of booleans that are used to select which instances are used for train and validation 1

    list_val_idx2 : [(array_bool_train, array_bool_val), ...]
        Arrays of booleans that are used to select which instances are used for train and validation 2

    prev_pred : [(node, idx, y_hat), ...]
        Predictions of the predictors of the ancestor nodes

    level : int
        Depth of the node
    """

    def __init__(self, X, y, tops, list_val_idx1, list_val_idx2, prev_pred, level=0):
        self.X = X
        self.y = y
        self.tops = tops
        self.list_val_idx1 = list_val_idx1
        self.list_val_idx2 = list_val_idx2
        if len(self.list_val_idx2) != 1:
            raise ValueError('The list_val_idx2 must have only one training and validation set')
        self.prev_pred = prev_pred
        self.level = level
        # Node from which it takes the best predictor
        self.node_predictor = None
        # Best predictor trained with the data in this node
        self.idx_best_predictor = None
        self.list_predictors = []
        self.goodness1 = None
        self.goodness1_display = None
        self.goodness2 = None
        self.sons = []
        self.decision_fun = None
        # self.aggr_fun = None
        self.w = None
        self.n_inst = X.shape[0]

        # self._check_val_idx()
        # A bitmap (array of bool) expressing all the instances used for validation
        self.all_val_bitmap1 = np.sum([val_bitmap for _, val_bitmap in self.list_val_idx1], axis=0) > 0
        self.all_val_bitmap2 = np.sum([val_bitmap for _, val_bitmap in self.list_val_idx2], axis=0) > 0
        self.n_inst_val1 = self.all_val_bitmap1.sum()
        self.n_inst_val2 = self.all_val_bitmap2.sum()

    def fit(self):
        """
        Fits the best predictor for this node. Uses the self.list_val_idx to evaluate the goodness of each predictor.
        """
        self.tops.timers.start('fit_node')
        self.tops.timers.start('init_fit_node')
        # All the predictions made at this node using all the available predictors (initialized to nan)
        node_all_pred_y = [np.zeros(len(self.y)) + np.nan for _ in range(len(self.tops.predictors_wrapped))]
        self.tops.timers.stop('init_fit_node')

        self.list_predictors = [copy.deepcopy(predictor) for predictor in self.tops.predictors_wrapped]

        for train_bitmap, val_bitmap in self.list_val_idx1:
            self.tops.timers.start('train_val')
            X_train = self.X[train_bitmap]
            X_val = self.X[val_bitmap]
            y_train = self.y[train_bitmap]
            self.tops.timers.stop('train_val')

            for i, my_predictor in enumerate(self.list_predictors):
                self.tops.timers.start('fit_base')
                my_predictor.fit(X_train, y_train)
                self.tops.timers.stop('fit_base')
                self.tops.timers.start('predict_base')
                y_pred = my_predictor.predict(X_val)
                self.tops.timers.stop('predict_base')
                node_all_pred_y[i][val_bitmap] = y_pred

        # evaluate best goodness
        self.tops.timers.start('eval_goodness1')
        list_node_goodness = [self.tops.goodness_metric(self.y[self.all_val_bitmap1], y_pred[self.all_val_bitmap1])
                              for y_pred in node_all_pred_y]
        best_node_goodness = max(list_node_goodness)

        list_prev_goodness = [(node, idx, self.tops.goodness_metric(self.y[self.all_val_bitmap1], y_pred[self.all_val_bitmap1]))
                              for node, idx, y_pred in self.prev_pred]
        best_parent_node, idx_best_pred, best_prev_goodness = max(list_prev_goodness, key=lambda x: x[2], default=(None, 0, -np.inf))
        self.tops.timers.stop('eval_goodness1')

        # use the classifier in that node
        if best_node_goodness > best_prev_goodness:
            self.idx_best_predictor = list_node_goodness.index(best_node_goodness)
            # the node that contains the best predictor for this node is itself
            self.node_predictor = self
            self.goodness1 = best_node_goodness * self.all_val_bitmap1.sum()
            self.goodness1_display = best_node_goodness
            # Update prev_pred
            self.prev_pred.extend((self, idx, node_all_pred_y[idx]) for idx in range(len(self.list_predictors)))

        # use the classifier from a parent node
        else:
            self.node_predictor = best_parent_node
            self.idx_best_predictor = idx_best_pred
            self.goodness1 = best_prev_goodness * self.all_val_bitmap1.sum()
            self.goodness1_display = best_prev_goodness
        self.tops.timers.stop('fit_node')

    def split(self):
        """
        Finds the split that gives the best goodness for the generated sons. If the goodness of the sons improve the
        one from the parent node, the split continues recursively. Otherwise, it stops the recursion.
        """
        # retrain with the training and V1 data
        for predictor in self.list_predictors:
            idx = self.list_val_idx2[0][0]
            predictor.fit(self.X[idx], self.y[idx])

        # stop criteria
        if self.X.shape[0] < self.tops.min_inst or len(unique_labels(self.y)) == 1:
            return

        self.tops.timers.start('split_node')
        list_best_sons = [self.DummyNode()]
        best_decision_fun = None
        for col in range(self.X.shape[1]):
            # Conditions to stop spliting with this feature

            if self.tops.metadata[col]['type'] == ToPs.categorical:
                # We only do the split with the first column representing a modality of a categorical feature
                if col != self.tops.metadata[col]['idx_modalities'][0]:
                    continue
            self.tops.timers.start('get_split')
            list_splits = self.tops.split_type.get_split(self.X, self.y, col, self.tops.metadata)
            self.tops.timers.stop('get_split')
            for decision_fun, n_bins in list_splits:
                # We cannot continue spliting since all the instances will go to the same bin
                if n_bins == 1:
                    continue

                X_idx = decision_fun(self.X)
                list_sons = []
                for i in range(n_bins):
                    # here I am creating deep copies
                    belongs_to_son = X_idx == i
                    # if the number of variables is high this step can be costly
                    self.tops.timers.start('x_y_son')
                    X_son = self.X[belongs_to_son]
                    y_son = self.y[belongs_to_son]
                    self.tops.timers.stop('x_y_son')
                    # we don't create a new tentative son if it doesn't contain any data
                    if len(y_son) == 0:
                        continue

                    self.tops.timers.start('idx_val_son')
                    list_val_idx1_son = [(train_bitmap[belongs_to_son], val_bitmap[belongs_to_son])
                                         for train_bitmap, val_bitmap in self.list_val_idx1]
                    list_val_idx2_son = [(train_bitmap[belongs_to_son], val_bitmap[belongs_to_son])
                                         for train_bitmap, val_bitmap in self.list_val_idx2]

                    # TODO Parametrize the minimum allowed data for the validations
                    not_enough_train1 = any(train_bitmap.sum() == 0 for train_bitmap, _ in list_val_idx1_son)
                    not_enough_train2 = any(train_bitmap.sum() == 0 for train_bitmap, _ in list_val_idx2_son)
                    not_enough_v1 = sum(val_bitmap.sum() for _, val_bitmap in list_val_idx1_son) < self.tops.min_inst_val
                    not_enough_v2 = sum(val_bitmap.sum() for _, val_bitmap in list_val_idx2_son) < self.tops.min_inst_val
                    self.tops.timers.stop('idx_val_son')

                    if not_enough_train1 or not_enough_train2 or not_enough_v1 or not_enough_v2:
                        dummy_son = self.DummyNode()
                        list_sons = [dummy_son]
                        break
                    # calculate in which son the previous predictions will go
                    prev_pred_son = [(node, idx, pred[belongs_to_son]) for node, idx, pred in self.prev_pred]

                    son = Node(X_son, y_son, self.tops, list_val_idx1_son, list_val_idx2_son,
                               prev_pred_son, self.level + 1)
                    son.fit()
                    list_sons.append(son)

                # Update the list of best sons found until now
                goodness_sons = sum((son.goodness1 for son in list_sons))
                goodness_best_sons = sum((son.goodness1 for son in list_best_sons))
                if goodness_sons > goodness_best_sons:
                    list_best_sons = list_sons
                    best_decision_fun = decision_fun

        goodness_best_sons = sum((son.goodness1 for son in list_best_sons))
        if self._hoeffding_criteria(goodness_best_sons):
            self.sons = list_best_sons
            self.decision_fun = best_decision_fun

            for son in self.sons:
                son.split()
        self.tops.timers.stop('split_node')

    def _hoeffding_criteria(self, goodness_sons):
        goodness_sons = goodness_sons / self.n_inst_val1 + 0.000001
        goodness_parent = self.goodness1 / self.n_inst_val1 + 0.000001
        epsilon = goodness_sons - goodness_parent
        if epsilon <= 0:
            return False

        if 1 - goodness_parent > 1/6:
            # hoeffding
            delta1 = np.exp(-2 * epsilon**2 * self.n_inst_val1)
            delta2 = delta1
        else:
            # chernoff
            delta1 = np.exp(- epsilon**2 * self.n_inst_val1 / (3 * (1 - goodness_parent)))
            delta2 = np.exp(- epsilon**2 * self.n_inst_val1 / (3 * (1 - goodness_sons)))
        prob_son_better_than_parent = (1 - delta1/2) * (1 - delta2/2)
        return prob_son_better_than_parent > self.tops.min_prob_hoeffding


    def aggregate(self, list_prev_y_prob):
        """Creates an ensemble for each path from the root to a leaf. The ensemble is created by aggregating
        the predictors found along the path using a weighted average

        :param list_prev_y_prob: List of probabilities calculated in the previous nodes
        """
        self.tops.timers.start('aggregate_node')
        # A bitmap (array of bool) expressing all the instances used for validation
        all_val_bitmap = np.sum([val_bitmap for _, val_bitmap in self.list_val_idx2], axis=0) > 0
        # Probabilities calculated in this node
        y_pred_prob_local = np.zeros((len(self.y), self.tops.n_classes))

        for train_bitmap, val_bitmap in self.list_val_idx2:
            # X_train = self.X[train_bitmap]
            X_val = self.X[val_bitmap]
            # y_train = self.y[train_bitmap]

            best_pred = self.node_predictor.list_predictors[self.idx_best_predictor]
            # best_pred.fit(X_train, y_train)
            prob = best_pred.predict_proba(X_val)
            for i, clss in enumerate(best_pred.classes_):
                # TODO predict for regression and predict_proba for classification
                y_pred_prob_local[val_bitmap, clss] = prob[:, i]

        list_prev_y_prob.append((self, y_pred_prob_local))
        # Calculate the goodness over the V2 set. Only for informative purposes
        self.goodness2 = self.tops.goodness_metric(self.y[all_val_bitmap],
                                                   y_pred_prob_local[all_val_bitmap].argmax(axis=1))

        for predictor in self.list_predictors:
            # Train the predictors with all the available data
            predictor.fit(self.X, self.y)

        if len(self.sons) > 0:
            X_idx = self.decision_fun(self.X)
            for i in range(len(self.sons)):
                idx_son = X_idx == i
                list_prev_y_prob_son = [(node, node_y_pred[idx_son]) for node, node_y_pred in list_prev_y_prob]
                self.sons[i].aggregate(list_prev_y_prob_son)
        else:
            # Calculate the aggregation function
            list_y_prob = [y_prob[all_val_bitmap] for _, y_prob in list_prev_y_prob]
            y_true = self.y[all_val_bitmap]
            self._calculate_aggr_fun(list_y_prob, y_true)

        # At the end of the aggregation process, check the node is consistent
        self._check_node()
        self.tops.timers.stop('aggregate_node')

    def _calculate_aggr_fun(self, list_y_prob, y_true):
        n_inst = list_y_prob[0].shape[0]
        n_clss = list_y_prob[0].shape[1]
        mask = np.array([list(range(n_clss))] * n_inst) == np.reshape(y_true, (n_inst, 1))

        len_w = len(list_y_prob)
        # This restriction is the same as sum(w) = 1
        # constr = LinearConstraint(np.ones((1, len_w)), 1, 1)
        constr = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        # Initial guess for w
        w0 = np.ones(len_w) / len_w
        # The individual w has to be between 0 and 1
        bounds = Bounds(0, 1)
        # TODO Add gradient function
        # Objective function (minimization) // only for accuracy
        c = np.array([y_prob[mask] for y_prob in list_y_prob]).transpose() + 0.0001
        res = minimize(partial(self._min_fun, c), w0, constraints=constr, bounds=bounds)
        self.w = res['x']

    @staticmethod
    def _min_fun(c, w):
        aggr_prob = c.dot(w)
        log_prob = np.log(aggr_prob)
        # Tambe podria utilitzar l'arrel cubica centrant el 0.5 a 0
        return -log_prob.sum()

    def predict_proba(self, X, list_y_prob):
        """
        :param X: (n_inst x n_vars) Dataset to predict
        :param list_y_prob: [np.array_prev_pred, ...] Predictions from ancestor nodes
        :return:
        """
        y_pred_prob_local = np.zeros((X.shape[0], self.tops.n_classes))
        y_pred_prob_aux = self.node_predictor.list_predictors[self.idx_best_predictor].predict_proba(X)
        for i, clss in enumerate(self.node_predictor.list_predictors[self.idx_best_predictor].classes_):
            y_pred_prob_local[:, clss] = y_pred_prob_aux[:, i]
        list_y_prob.append(y_pred_prob_local)

        # This is a terminal node
        if self.w is not None:
            aggr_prob = np.dot(np.array(list_y_prob).transpose(), self.w).transpose()
            # return aggr_prob.argmax(axis=1)
            return aggr_prob

        # This is not a terminal node
        # y_pred could be passed by parameter and modified inside the function (a little bit more efficient)
        y_pred = np.zeros((X.shape[0], self.tops.n_classes))
        X_idx = self.decision_fun(X)
        for i in range(len(self.sons)):
            idx_son = X_idx == i
            X_son = X[idx_son]
            if X_son.shape[0] > 0:
                list_y_pred_son = [y_prev_pred[idx_son] for y_prev_pred in list_y_prob]
                y_pred_son = self.sons[i].predict_proba(X_son, list_y_pred_son)
                y_pred[idx_son] = y_pred_son
        return y_pred

    def _is_leaf(self):
        return len(self.sons) == 0

    def _check_node(self):
        if self._is_leaf():
            assert len(self.sons) == 0 and len(self.w) == (self.level + 1) and self.decision_fun is None
        else:
            assert len(self.sons) > 1 and self.w is None and self.decision_fun is not None

        assert len(self.list_predictors) > 0
        assert self.node_predictor is not None
        assert 0 <= self.idx_best_predictor < len(self.node_predictor.list_predictors)

    def clear_data(self):
        self.X = None
        self.y = None
        self.all_val_bitmap1 = None
        self.all_val_bitmap2 = None
        for son in self.sons:
            son.clear_data()

    def __str__(self):
        # if the node is a leaf, print also the weights of the path
        if self.w is not None:
            weights = self.level * '\t' + 'Weights: ' + str(self.w) + '\n'
        else:
            weights = ''
        predictor = self.node_predictor.list_predictors[self.node_predictor.idx_best_predictor]

        if self.decision_fun is not None:
            split_var_str = self.level * '\t' + 'Split var: ' + str(self.decision_fun.idx_var) + '\n'
        else:
            split_var_str = ''

        str_tree = 'nInst: ' + str(self.n_inst) + '\n' + \
                   self.level * '\t' + 'Predictor: ' + str(predictor)[:30] + '\n' + \
                   split_var_str + \
                   self.level * '\t' + 'From node: ' + str(self.node_predictor.level) + '\n' + \
                   self.level * '\t' + 'Goodness1: ' + str(self.goodness1_display) + '\n' + \
                   self.level * '\t' + 'Goodness2: ' + str(self.goodness2) + '\n' + \
                   weights

        for i in range(len(self.sons)):
            str_tree += (self.level + 1) * '\t' + str(i) + ' -> ' + self.sons[i].__str__()
        return str_tree

    class DummyNode:
        def __init__(self):
            self.goodness1 = -np.inf


class WrapperPredictor(BaseEstimator):
    """
    Class to wrap a predictor
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        if len(self.classes_) > 1:
            self.predictor.fit(X, y)
            # assert all(self.classes == self.predictor.classes_)
        return self

    def predict(self, X):
        if len(self.classes_) == 1:
            return np.ones(X.shape[0]) * self.classes_[0]
        else:
            return self.predictor.predict(X)

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            aux_prob = np.ones((X.shape[0], 1))
            return aux_prob
        else:
            return self.predictor.predict_proba(X)

    def __str__(self):
        return self.predictor.__str__()