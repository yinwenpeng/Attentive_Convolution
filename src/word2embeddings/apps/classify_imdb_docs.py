# -*- coding: utf-8 -*-
"""
Example usage:
X:\sa\corpora\imdb\txt_sentoken
X:\sa\embeddings\vlbl\wikipedia_small-general-nce5-960\vLBL.vocab
X:\sa\embeddings\vlbl\wikipedia_small-general-nce5-960\vLBL_960_14-03-30_23-07-11.embeddings_q
.
"""

from argparse import ArgumentParser
import json
from logging import getLogger
import os
import sys

from scipy.io import mmread
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
from sklearn.svm import LinearSVC

from cis.deep.utils import logger_config, file_line_generator, utf8_file_open
import numpy as np
from cis.deep.utils.embeddings import read_vocabulary_id_file, \
    compute_avg_text_embedding
import itertools
from cis.deep.utils.classification import calc_metrics


NO_OF_FOLDS = 10

log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description='Perform a 10-fold cross validation on the polarity ' +
        'dataset v2.0 of [PanLee04].')
parser.add_argument('corpus_dir',
        help='location of the pos and neg directories of the dataset')
parser.add_argument('vocabulary',
        help='Vocabulary file that contains list of tokens.')
parser.add_argument('embeddings',
        help='File that contains the trained word embeddings')
parser.add_argument('output_dir',
        help='directory to store the results in')

def convert_doc(doc, vocab, embs):
    """Convert the given document into a document vector.

    Average all word vectors to a final document vector.

    Parameters
    ----------
    doc : str
        filename of the document
    vocab : dict(str, int)
        id vocabulary
    embs : ndarray
        embeddings
    """

    with utf8_file_open(doc) as f:
        s = f.read()
        return compute_avg_text_embedding(s, vocab, embs)

def do_cross_validation(features, labels):
    """Perform the k-fold cross validation.

    Perform the k-fold cross validation, collect the result and return the
    single test instance predictions, as well as the classification results for
    each single fold and for the combination of all folds.

    Keyword arguments:
    features -- all features
    labels -- all labels
    classifier -- code of the classifier to create (see command line arguments)
    """
    single_predictions = []  # Store each single classification decision
    # Store the feature weights after the training
    weight_vectors = np.zeros((NO_OF_FOLDS, len(features.values()[0])))
    # Store classification results for each fold and for the entire task (i.e.,
    # entire cross validation).
    classification_result = np.zeros((NO_OF_FOLDS + 1, 5))

    for cur_fold, (train_names, test_names) in enumerate(imdb_cross_folds(features.keys())):
        train_data = [features[n] for n in train_names]
        train_labels = [labels[n] for n in train_names]
        model = train_model(train_data, train_labels)

        test_data = [features[n] for n in test_names]
        pred_labels = model.predict(test_data)
        true_labels = []

        for i in xrange(len(test_data)):
            single_predictions.append([cur_fold, test_names[i],
                    labels[test_names[i]], pred_labels[i]])
            true_labels.append(labels[test_names[i]])

        classification_result[cur_fold, :] = get_classification_result(cur_fold,
                true_labels, pred_labels)

        weight_vectors[cur_fold, :] = model.coef_

    return single_predictions, classification_result, weight_vectors

def get_classification_result(fold_no, true_labels, pred_labels):
    """Return classification resuls for one fold.

    Return an array containing accuracy, precision, recall, and f1, based on the
    given true and predicted labels.

    Parameters
    ----------
    fold_no : int
        this fold's number
    true_labels list(int)
        true labels
    pred_labels list(int)
        predicted labels

    Returns
    -------
    ndarray
        [fold number, accuracy, precision, recall, f1]
    """
    res = calc_metrics(true_labels, pred_labels)
    return np.asarray([fold_no] + [r for r in res])

def imdb_cross_folds(filenames):
    """Get the docs for training and testing to be used in a 10-fold x
    validation.

    Parameters
    ----------
    filenames : list(str)
        filenames of imdb docs; they contain the fold number 

    Returns
    -------
    list(str)
        names of training documents
    list(str)
        names of test documents
    """

    for i in xrange(10):
        test = filter(lambda f: f.startswith(u'cv' + unicode(i)), filenames)
        training = filter(lambda f: not f.startswith(u'cv' + unicode(i)), filenames)
        yield (training, test)

    raise StopIteration()

def load_data(corpus_dir, vocab, embs):
    """Load feature data and labels.

    Loads the documents from the imdb corpus and converts them into one feature
    vector per document by averaging the word representations of the text.

    Parameters
    ----------
    corpus_dir : str
        location of the dataset
    vocab : dict(str, int)
        id vocabulary
    embs : ndarray(m*n)
        word embeddings

    Returns
    -------
    features : dict(str, ndarray)
        map from a document name its document representations, which is the
        averaged word vectors
    labels : dict(str, int)
        map from a document name its label
    """
    pos_docs = os.listdir(os.path.join(corpus_dir, u'pos'))
    num_pos_docs = len(pos_docs)
    pos_docs = [os.path.join(corpus_dir, u'pos/', d) for d in pos_docs]
    neg_docs = os.listdir(os.path.join(corpus_dir, u'neg'))
    neg_docs = [os.path.join(corpus_dir, u'neg/', d) for d in neg_docs]
    docs = pos_docs + neg_docs
    features = dict()
    labels = dict()

    for (count, d) in enumerate(docs):
        basename = os.path.basename(d)
        features[basename] = convert_doc(d, vocab, embs)
        labels[basename] = 1 if count < num_pos_docs else 0

    return features, labels

def train_model(features, labels):
    """Create, train, and return a model using the given features and labels.

    Parameters
    ----------
    features : list(ndarray)
        features of training instances
    labels : list(int)
        labels of training instances
    """
    model = LinearSVC()
    model.fit(features, labels)
    return model

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading embeddings')
    vocab = read_vocabulary_id_file(args.vocabulary)
    embs = np.loadtxt(args.embeddings)

    log.info('loading documents')
    features, labels = load_data(args.corpus_dir, vocab, embs)

    log.info('performing cross validation')
    single_predictions, classification_result, weight_vectors = \
            do_cross_validation(features, labels)

    log.info('storing results')
    np.savetxt(os.path.join(args.output_dir, 'svm-weights.csv'),
            weight_vectors, '%f', ';', '\n')

    with utf8_file_open(os.path.join(args.output_dir, 'predictions.csv'), 'w') \
            as pred_file:
        pred_file.write(u'fold_no;doc;true_label;pred_label\n')

        for sp in single_predictions:
            pred_file.write(u';'.join(map(unicode, sp)) + u'\n')

    all_true_labels = [sp[2] for sp in single_predictions]
    all_pred_labels = [sp[3] for sp in single_predictions]
    confusion = confusion_matrix(all_true_labels, all_pred_labels)

    np.savetxt(os.path.join(args.output_dir, 'confusion_matrix.csv'),
            confusion, '%d', ';', '\n')

    classification_result[NO_OF_FOLDS, :] = get_classification_result(-1,
                all_true_labels, all_pred_labels)

    header = u'fold_no;accuracy;precision;recall;f1'
    np.savetxt(os.path.join(args.output_dir, 'metrics.csv'),
            classification_result, '%f', u';', u'\n', header=header)

    log.info(classification_result)
    log.info('finished')

if __name__ == '__main__':
    sys.exit(main())
