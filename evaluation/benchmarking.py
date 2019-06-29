# coding=utf-8
from tops.tops import ToPs, SplitImpurity, SplitOriginal, is_numeric
from sklearn.model_selection import cross_validate
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from scipy.io import arff
import os
import pandas as pd
import time
from datetime import datetime
import numpy as np
import multiprocessing
import psutil
import traceback


class myCVProcess(multiprocessing.Process):
    def __init__(self, child_con, predictor, X, y, n_folds):
        multiprocessing.Process.__init__(self)
        self.child_con = child_con
        self.predictor = predictor
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.res = None

    def run(self):
        roc_auc_scorer = make_scorer(self.my_roc_auc_scorer, needs_proba=True)
        self.X, self.y = shuffle(self.X, self.y)
        try:
            res = cross_validate(self.predictor, self.X, self.y,
                                 scoring={'accuracy': 'accuracy', 'precision_weighted': 'precision_weighted',
                                          'recall_weighted': 'recall_weighted', 'roc_auc_weighted': roc_auc_scorer,
                                          'f1_weighted': 'f1_weighted'},
                                 cv=self.n_folds, return_train_score=False, n_jobs=-1, error_score='raise')
        except:
            res = None
            traceback.print_exc()
        self.child_con.send(res)

    def terminate(self):
        parent = psutil.Process(self.pid)
        for child in parent.children(recursive=True):
            child.kill()
        multiprocessing.Process.terminate(self)

    @staticmethod
    def my_roc_auc_scorer(y_true, y_score):
        labels = np.unique(y_true)
        total_roc_auc = 0
        try:
            for i, label in enumerate(labels):
                y_label_true = (y_true == label) + 0
                y_label_prob = y_score[:, i]
                total_roc_auc += (sum(y_true == label) / len(y_true)) * roc_auc_score(y_label_true, y_label_prob)
        except:
            total_roc_auc = np.nan
        return total_roc_auc


def get_X_y(df, dummies):
    # Guess which column represents the class
    if 'class' in df.columns:
        clss = 'class'
    elif 'Class' in df.columns:
        clss = 'Class'
    elif 'CLASS' in df.columns:
        clss = 'CLASS'
    elif 'y' in df.columns:
        clss = 'y'
    else:
        clss = df.columns[-1]

    if dummies:
        X = pd.get_dummies(df.drop(columns=clss)).values
    else:
        X = df.drop(columns=clss).values
    y = df[clss].values

    return X, y


def load_arff(path, dummies):
    data, metadata = arff.loadarff(path)
    df = pd.DataFrame(data)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(axis=1, inplace=True)
    for col in df.columns:
        if type(df[col][0]) is bytes:
            df[col] = df[col].str.decode("utf-8")

    return get_X_y(df, dummies)


def load_csv(path, dummies):
    df = pd.read_csv(path)
    return get_X_y(df, dummies)


def eliminate_rare_classes(X, y, min_n_inst):
    classes = np.unique(y)
    n_inst_clss = np.array([sum(y == clss) for clss in classes])
    idx_enought_inst = np.full(y.shape, False)
    for i in range(len(classes)):
        if n_inst_clss[i] >= min_n_inst:
            idx_enought_inst |= y == classes[i]

    return X[idx_enought_inst, :], y[idx_enought_inst]


def generate_metainfo(X, y):
    n_feat = X.shape[1]
    n_feat_num = sum(is_numeric(X[0, j]) for j in range(n_feat))
    n_feat_cat = n_feat - n_feat_num
    n_inst = X.shape[0]
    n_classs = len(unique_labels(y))

    return {'n_feat': n_feat, 'n_feat_num': n_feat_num, 'n_feat_cat': n_feat_cat, 'n_inst': n_inst,
            'n_classes': n_classs}


def create_out_file(out_file_prefix):
    if not os.path.exists('../out/'):
        os.mkdir('../out/')
    filename = '../out/' + out_file_prefix + '_rep' + str(n_rep) + '_CV' + str(n_folds) + '_' + \
               datetime.today().isoformat().replace('T', '_').replace(':', '-').split('.')[0] + '.csv'
    with open(filename, 'x') as f:
        # write header
        write_item(f, 'dataset')
        write_item(f, 'n_inst')
        write_item(f, 'n_feat')
        write_item(f, 'n_feat_num')
        write_item(f, 'n_feat_cat')
        write_item(f, 'n_classes')
        write_item(f, 'predictor')
        write_item(f, 'mean_accuracy')
        write_item(f, 'std_accuracy')
        write_item(f, 'mean_auc')
        write_item(f, 'std_auc')
        write_item(f, 'mean_precision')
        write_item(f, 'std_precision')
        write_item(f, 'mean_recall')
        write_item(f, 'std_recall')
        write_item(f, 'mean_f1')
        write_item(f, 'std_f1')
        write_item(f, 'exec_time', last_item=True)
        f.write('\n')
    return filename


def write_item(f, item, last_item=False):
    f.write(str(item))
    if not last_item:
        f.write(';')


def write_result(filename, dataset_name, predictor, metainfo, res, ex_time):
    with open(filename, 'a') as f:
        write_item(f, dataset_name)
        write_item(f, metainfo['n_inst'])
        write_item(f, metainfo['n_feat'])
        write_item(f, metainfo['n_feat_num'])
        write_item(f, metainfo['n_feat_cat'])
        write_item(f, metainfo['n_classes'])
        write_item(f, ' '.join(predictor.__repr__().split()))
        write_item(f, res['test_accuracy'].mean())
        write_item(f, res['test_accuracy'].std())
        write_item(f, res['test_roc_auc_weighted'].mean())
        write_item(f, res['test_roc_auc_weighted'].std())
        write_item(f, res['test_precision_weighted'].mean())
        write_item(f, res['test_precision_weighted'].std())
        write_item(f, res['test_recall_weighted'].mean())
        write_item(f, res['test_recall_weighted'].std())
        write_item(f, res['test_f1_weighted'].std())
        write_item(f, res['test_f1_weighted'].std())
        write_item(f, ex_time, last_item=True)
        f.write('\n')


def write_predictors(filename, dataset_name, pred_name):
    with open(filename, 'a') as f:
        f.write('\n-------------------------------------------\n')
        f.write('-------------------------------------------\n')
        f.write('\n')
        f.write(pred_name + '\n')
        f.write(dataset_name + '\n\n')


def cv_timeout(predictor, X, y):
    # We want to be able to reproduce the results
    np.random.seed(0)

    total_res = {'test_accuracy': np.array([]), 'test_roc_auc_weighted': np.array([]),
                 'test_precision_weighted': np.array([]), 'test_recall_weighted': np.array([]),
                 'test_f1_weighted': np.array([])}
    total_time = 0
    # predictors_str = []

    # Repeat the CV 5 times, to have an stable result
    for _ in range(n_rep):
        parent_con, child_con = multiprocessing.Pipe()
        cv_process = myCVProcess(child_con, predictor, X, y, n_folds)
        cv_process.start()

        t = time.time()
        cv_process.join(timeout=timeout)
        total_time += time.time() - t

        if cv_process.is_alive():
            for key in total_res:
                total_res[key] = np.array([np.nan])
            cv_process.terminate()
            print("Timeout!!!")
            break
        else:
            res = parent_con.recv()
            # We receive a None if an exception occurred in the cv
            if res is None:
                for key in total_res:
                    total_res[key] = np.array([np.nan])
                total_time = 0
                break
            else:
                for key in total_res:
                    # We append the mean value of the cv
                    total_res[key] = np.append(total_res[key], res[key])
                # predictors_str.extend(res['estimator'])

    return total_res, total_time


def main(list_predictors, out_file_prefix, start_file='', files=None):
    out_filename = create_out_file(out_file_prefix)
    log_filename = out_filename.replace('.csv', '_predictor.txt')
    datasets_path = '../datasets/'
    for file in os.listdir(datasets_path):
        if file < 'c' or files is not None and file not in files:
            continue
        print('Loading dataset', file)
        if file.endswith('.arff') and file >= start_file:
            X, y = load_arff(datasets_path + '/' + file, True)
        elif file.endswith('.csv') and file >= start_file:
            X, y = load_csv(datasets_path + '/' + file, True)
        else:
            continue
            # raise ValueError("Cannot read the file " + file)

        X, y = eliminate_rare_classes(X, y, n_folds * 2)
        # At least 2 classes and 100 instances to do the training and evaluation
        if not (len(np.unique(y)) >= 2 and len(y) >= min_inst):
            continue
        metainfo = generate_metainfo(X, y)

        for predictor in list_predictors:
            predictor.file_log = log_filename
            print('CV dataset', file, 'with predictor', predictor.__repr__())
            write_predictors(log_filename, file, predictor.__repr__())
            res, ex_time = cv_timeout(predictor, X, y)
            write_result(out_filename, file, predictor, metainfo, res, ex_time)


if __name__ == '__main__':
    n_folds = 8
    n_rep = 1
    min_inst = 1000
    timeout = 5 * 60 * 60  # 5h for each CV

    lr = LogisticRegression(solver='liblinear', multi_class='auto')
    rf = RandomForestClassifier(n_estimators=500)
    dt = DecisionTreeClassifier()
    nb = GaussianNB()

    # main([lr], 'LR')
    # main([rf], 'RF')
    # main([dt], 'DT')
    # main([nb], 'NB')


    list_percentile = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # main([ToPs([LogisticRegression(solver='liblinear', multi_class='auto')], SplitOriginal(list_percentile),
    #            accuracy_score, 100, 0.0, 0.15, 0.1, 20)], 'ToPs_LR')
    # main([ToPs([DecisionTreeClassifier()], SplitOriginal(list_percentile),
    #            accuracy_score, 100, 0.0, 0.15, 0.1, 20)], 'ToPs_DT')
    # main([ToPs([GaussianNB()], SplitOriginal(list_percentile),
    #            accuracy_score, 100, 0.0, 0.15, 0.1, 20)], 'ToPs_NB')
    #
    # main([ToPs([LogisticRegression(solver='liblinear', multi_class='auto'),
    #             DecisionTreeClassifier(),
    #             GaussianNB()], SplitOriginal(list_percentile),
    #            accuracy_score, 100, 0.0, 0.15, 0.1, 20)], 'ToPs_full')

    # main([ToPs([LogisticRegression(solver='liblinear', multi_class='auto')], SplitOriginal(list_percentile),
    #            accuracy_score, 100, 0.8, 0.15, 0.1, 20)], 'ToPs_LR')
    # main([ToPs([DecisionTreeClassifier()], SplitOriginal(list_percentile),
    #            accuracy_score, 100, 0.8, 0.15, 0.1, 20)], 'ToPs_DT')
    # main([ToPs([GaussianNB()], SplitOriginal(list_percentile),
    #            accuracy_score, 100, 0.8, 0.15, 0.1, 20)], 'ToPs_NB')
    #
    # main([ToPs([LogisticRegression(solver='liblinear', multi_class='auto'),
    #             DecisionTreeClassifier(),
    #             GaussianNB()], SplitOriginal(list_percentile),
    #            accuracy_score, 100, 0.8, 0.15, 0.1, 20)], 'ToPs_full')
    #
    # main([ToPs([LogisticRegression(solver='liblinear', multi_class='auto')], SplitImpurity(SplitImpurity.gini, 300),
    #            accuracy_score, 100, 0.0, 0.15, 0.1, 20)], 'ToPs_LR')
    main([ToPs([DecisionTreeClassifier()], SplitImpurity(SplitImpurity.gini, 300),
               accuracy_score, 100, 0.0, 0.15, 0.1, 20)], 'ToPs_DT')
    main([ToPs([GaussianNB()], SplitImpurity(SplitImpurity.gini, 300),
               accuracy_score, 100, 0.0, 0.15, 0.1, 20)], 'ToPs_NB')

    main([ToPs([LogisticRegression(solver='liblinear', multi_class='auto'),
                DecisionTreeClassifier(),
                GaussianNB()], SplitImpurity(SplitImpurity.gini, 300),
               accuracy_score, 100, 0.0, 0.15, 0.1, 20)], 'ToPs_full')