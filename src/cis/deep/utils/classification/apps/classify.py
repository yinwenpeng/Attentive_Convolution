# -*- coding: utf-8 -*-
"""
"""

from argparse import ArgumentParser
from logging import getLogger
import os
import sys

from sklearn.metrics.metrics import confusion_matrix
from sklearn.svm import LinearSVC

from cis.deep.utils import logger_config, file_line_generator, \
    save_object_to_file
from cis.deep.utils.classification import calc_metrics
import numpy as np
from sklearn.dummy import DummyClassifier

# import pydevd
# pydevd.settrace(host='129.187.148.250', stdoutToServer=True,
#         stderrToServer=True)

log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description='Train and test a classifier.')
parser.add_argument('train_data',
        help="""File containing the features as dense matrix. bz2 and gz are
        supported.""")
parser.add_argument('train_labels',
        help="""File containing the data labels. One label per line.""")
parser.add_argument('test_data',
        help="""File containing the features as dense matrix. bz2 and gz are
        supported.""")
parser.add_argument('test_labels',
        help="""File containing the data labels. One label per line.""")
parser.add_argument('output_dir',
        help='directory to store the results in')

parser.add_argument('-n', '--normalize', action='store_true',
        help="""Normalize each feature to zero mean and 1 std dev. That makes
        sense if the the values of different features are very different.""")
parser.add_argument('-m', '--mode', action='store_true',
        help="""compute the results using mode, i.e., the majority class of the
        training data.""")

def get_classification_result(true_labels, pred_labels):
    """Return classification resuls for one fold.

    Return an array containing accuracy, precision, recall, and f1, based on the
    given true and predicted labels.

    Keyword arguments:
    fold_no -- this fold's number
    true_labels -- true labels
    pred_labels -- predicted labels
    """
    res = np.zeros((1, 4))
    res[:] = calc_metrics(true_labels, pred_labels)
    return res

def calc_results(train_features, train_labels, test_features, test_labels,
        normalize=False, mode=False):
    """Perform the k-fold cross validation.

    Perform the k-fold cross validation, collect the result and return the
    single test instance predictions, as well as the classification results for
    each single fold and for the combination of all folds.

    Keyword arguments:
    train_features -- all train_features
    train_labels -- all train_labels
    normalize -- normalize features to have zero mean and 1 std dev
    mode -- use mode (majority label) instead of liblinear
    """

    if normalize and not mode:
        # compute the mean and std dev only on the training data, but also
        # apply it to the test data.
        mean = np.mean(train_features, axis=0)
        std_dev = np.std(train_features, axis=0, dtype=float)
        train_features = (train_features - mean) / std_dev
        test_features = (test_features - mean) / std_dev

    if mode:
        model = model = DummyClassifier(strategy='most_frequent')
    else:
        model = LinearSVC(random_state=84447)

    model.fit(train_features, train_labels)
    pred_labels = model.predict(test_features)

    single_predictions = np.transpose(np.vstack((xrange(test_labels.shape[0]),
            test_labels, pred_labels)))

    classification_result = get_classification_result(test_labels, pred_labels)

    if mode:
        weight_vectors = model.class_prior_
    else:
        # Store the feature weights after the training
        weight_vectors = model.coef_

    return single_predictions, classification_result, weight_vectors, model

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading feature and label data')
    train_labels = np.asarray(map(int, list(file_line_generator(args.train_labels))))
    train_features = np.loadtxt(args.train_data)

    if train_features.ndim == 1:
        train_features = train_features.reshape((train_features.shape[0], 1))

    test_labels = np.asarray(map(int, list(file_line_generator(args.test_labels))))
    test_features = np.loadtxt(args.test_data)

    if test_features.ndim == 1:
        test_features = test_features.reshape((test_features.shape[0], 1))

    log.info('performing classification')
    single_predictions, classification_result, weight_vectors, model = \
            calc_results(train_features, train_labels, test_features,
            test_labels, args.normalize, args.mode == True)

    log.info('storing results')
    save_object_to_file(model, os.path.join(args.output_dir, 'svm'))

    np.savetxt(os.path.join(args.output_dir, 'weights.csv'),
            weight_vectors, '%f', ';', '\n')

    header = 'instance_index;true_label;pred_label'
    np.savetxt(os.path.join(args.output_dir, 'predictions.csv'),
            single_predictions, '%d', ';', '\n', header=header)

    all_true_labels = single_predictions[:, 1]
    all_pred_labels = single_predictions[:, 2]
    confusion = confusion_matrix(all_true_labels, all_pred_labels)

    np.savetxt(os.path.join(args.output_dir, 'confusion_matrix.csv'),
            confusion, '%d', ';', '\n')

    header = 'accuracy;precision;recall;f1'
    np.savetxt(os.path.join(args.output_dir, 'metrics.csv'),
            classification_result, '%f', ';', '\n', header=header)

    log.info(classification_result)
    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
