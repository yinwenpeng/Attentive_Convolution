# -*- coding: utf-8 -*-
"""
example usage:
-n
X:\sa\experiments\contextual_polarity\vlbl\sentiment-wnd3_3-nce5\classification\1ep\distrib.out
X:\sa\experiments\contextual_polarity\vlbl\sentiment-wnd3_3-nce5\classification\ebert,20140515-label
.
"""

from argparse import ArgumentParser
from logging import getLogger
import os
import sys

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics.metrics import confusion_matrix
from sklearn.svm import LinearSVC

from cis.deep.utils import logger_config, file_line_generator
from cis.deep.utils.classification import calc_metrics
import numpy as np


# from sklearn.dummy import DummyClassifier
log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description='Perform a 10-fold cross validation on given feature data.')
parser.add_argument('feature_file',
        help="""File containing the features as dense matrix. bz2 and gz are
        supported.""")
parser.add_argument('label_file',
        help="""File containing the data labels. One label per line.""")
parser.add_argument('output_dir',
        help='directory to store the results in')

parser.add_argument('-n', '--normalize', action='store_true',
        help="""Normalize each feature to zero mean and 1 std dev. That makes
        sense if the the values of different features are very different.""")

NO_OF_FOLDS = 10

def get_classification_result(fold_no, true_labels, pred_labels):
    """Return classification resuls for one fold.

    Return an array containing accuracy, precision, recall, and f1, based on the
    given true and predicted labels.

    Keyword arguments:
    fold_no -- this fold's number
    true_labels -- true labels
    pred_labels -- predicted labels
    """
    res = np.zeros(5)
    res[0] = fold_no

    acc, prec, rec, f1 = calc_metrics(true_labels, pred_labels)
    res[1:5] = [acc, prec, rec, f1]
    return res

def calc_results(train_features, train_labels, normalize=False):
    """Perform the k-fold cross validation.

    Perform the k-fold cross validation, collect the result and return the
    single test instance predictions, as well as the classification results for
    each single fold and for the combination of all folds.

    Keyword arguments:
    train_features -- all train_features
    train_labels -- all train_labels
    """
    skf = StratifiedKFold(train_labels, NO_OF_FOLDS)
    single_predictions = []  # Store each single classification decision
    # Store the feature weights after the training
    weight_vectors = np.zeros((NO_OF_FOLDS, train_features.shape[1]))

    # Store classification results for each fold and for the entire task (i.e.,
    # entire cross validation).
    classification_result = np.zeros((NO_OF_FOLDS + 1, 5))

    for cur_fold, (train_idx, test_idx) in enumerate(skf):
        train_data = train_features[train_idx]
        test_data = train_features[test_idx]

        if normalize:
            # compute the mean and std dev only on the training data, but also
            # apply it to the test data.
            mean = np.mean(train_features[train_idx, :], axis=0)
            std_dev = np.std(train_features[train_idx, :], axis=0, dtype=float)
            train_data = (train_data - mean) / std_dev
            test_data = (test_data - mean) / std_dev

        model = LinearSVC(random_state=84447)
        model.fit(train_data, train_labels[train_idx])
        pred_labels = model.predict(test_data)

        fold_array = np.empty(test_idx.shape[0])
        fold_array.fill(cur_fold)
        single_predictions.append(np.transpose(np.vstack((fold_array, test_idx,
                train_labels[test_idx], pred_labels))))
        classification_result[cur_fold, :] = get_classification_result(cur_fold,
                train_labels[test_idx], pred_labels)
        weight_vectors[cur_fold, :] = model.coef_

    single_predictions = np.vstack(single_predictions)
    return single_predictions, classification_result, weight_vectors

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading feature and label data')
    labels = np.asarray(map(int, list(file_line_generator(args.label_file))))
    features = np.loadtxt(args.feature_file)

    log.info('performing cross validation')
    single_predictions, classification_result, weight_vectors = \
            calc_results(features, labels, args.normalize)

    log.info('storing results')
    np.savetxt(os.path.join(args.output_dir, 'svm-weights.csv'),
            weight_vectors, '%f', ';', '\n')

    header = 'fold_no;instance_index;true_label;pred_label'
    np.savetxt(os.path.join(args.output_dir, 'predictions.csv'),
            single_predictions, '%d', ';', '\n', header=header)

    all_true_labels = single_predictions[:, 2]
    all_pred_labels = single_predictions[:, 3]
    confusion = confusion_matrix(all_true_labels, all_pred_labels)

    np.savetxt(os.path.join(args.output_dir, 'confusion_matrix.csv'),
            confusion, '%d', ';', '\n')

    classification_result[NO_OF_FOLDS, :] = get_classification_result(-1,
                all_true_labels, all_pred_labels)

    header = 'fold_no;accuracy;precision;recall;f1'
    np.savetxt(os.path.join(args.output_dir, 'metrics.csv'),
            classification_result, '%f', ';', '\n', header=header)

    log.info(classification_result)
    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
