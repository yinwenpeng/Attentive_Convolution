# -*- coding: utf-8 -*-
"""
example usage:
-p -r -f
X:\sa\experiments\contextual_polarity\mlp\sent_1\amazon\sanity_test-most_well-binary\features-predict-tmp
X:\sa\experiments\contextual_polarity\mlp\sent_1\amazon\sanity_test-most_well-binary\features-predict-out-cleaned
"""
from argparse import ArgumentParser
from logging import getLogger
import sys

from sklearn.metrics.metrics import accuracy_score, \
    precision_recall_fscore_support

from cis.deep.utils import logger_config, file_line_generator


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(description="""Calculate the error metrics accuracy,
        precision, recall, and f-measure for the given true and predicted
        labels. Labels must be numeric type. This application is a wrapper
        for sklearn.metrics.accuracy_score and
        sklearn.metrics.precision_recall_fscore_support. Look up their
        documentation to find the explanations of the parameters.""")
parser.add_argument('true_labels', help='true labels, one per line')
parser.add_argument('pred_labels', help='predicted labels, one per line')

parser.add_argument('-p', '--precision', action='store_true',
        help='calculate precision')
parser.add_argument('-r', '--recall', action='store_true',
        help='calculate recall')
parser.add_argument('-f', '--f_measure', action='store_true',
        help='calculate f-measure')

parser.add_argument('-b', '--beta', default=1.0, type=float,
        help='beta value of f-measure')
parser.add_argument('-o', '--pos_label', default='1',
        help='label of the positive class in a binary classification task')
parser.add_argument('-a', '--avg', choices=['none', 'micro', 'macro', 'samples',
        'weighted'], default='none',
        help='label of the positive class in a binary classification task')

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')
    true = []
    pred = []

    for line in file_line_generator(args.true_labels):
        true.append(line)

    for line in file_line_generator(args.pred_labels):
        pred.append(line)

    acc = accuracy_score(true, pred)
    log.info('accuracy: %f' % acc)

    if args.precision or args.recall or args.f_measure:
        p, r, f, _ = precision_recall_fscore_support(true, pred, args.beta,
                pos_label=args.pos_label,
                average=None if not args.avg else args.avg)

        if args.precision:
            log.info('precision: %f' % p)
        if args.recall:
            log.info('recall: %f' % r)
        if args.f_measure:
            log.info('f-measure: %f' % f)

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
