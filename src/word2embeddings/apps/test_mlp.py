# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
Example usage:
X:\sa\experiments\contextual_polarity\mlp\easy_test\1-1-features-predict
X:\sa\experiments\contextual_polarity\mlp\easy_test\1-1-features-predict-out
MultiLayerPerceptron_1_13-12-05_18-20-37.model
"""

from argparse import ArgumentParser
from logging import getLogger
import logging
import sys


# CAUTION: remove the Theano path before importing any of my or Theano's
# libraries

# print '\n'
# local
# if 'C:\\Anaconda\\lib\\site-packages\\theano_test-current' in sys.path:
#     sys.path.remove('C:\\Anaconda\\lib\\site-packages\\theano_test-current')
#     print 'removed old theano_test path'
# # Calculus
# if '/usr/local/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg' in sys.path:
#     sys.path.remove('/usr/local/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg')
#     print 'removed old theano_test path'
# Omega
# if '/usr/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg' in sys.path:
#     sys.path.remove('/usr/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg')
#     print 'removed old theano_test path'
# sys.path.insert(0, '/mounts/Users/cisintern/ebert/data/promotion/src/deep/src/word2embeddings/main/resources/theano_develop')
# print '\n'.join(sys.path)
# print '\n'
# sys.path.remove('/usr/lib/python2.7/site-packages')
# exit()

from cis.deep.utils import utf8_file_open, logger_config
from word2embeddings.nn.predictor import MlpPredictor
from word2embeddings.tools.util import debug

log = getLogger(__name__)
logger_config(log)

from theano import version
print version.full_version

parser = ArgumentParser()
parser.add_argument('--disable-padding', dest='disable_padding',
        action='store_true', default=False,
        help='Disable padding sentences while generating examples')
parser.add_argument('--binary',
        action='store_true',
        help='Predict binary values, i.e., round output values to {0, 1}')

parser.add_argument('predict_file',
        help='Document with examples to predict the label of.')
parser.add_argument('result_file',
        help='Document to which the predictions will be written.')
parser.add_argument('load_model',
        help='Proceed training with the given model file.')

# Argument for MiniBatchTrainer
# parser.add_argument('--batch-size', dest='batch_size', type=int, default=16)


def main(argv=None):
    log.info('started application')

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args()
    log.info('start parameters: ' + str(args))

    if log.level == logging.DEBUG:
        sys.excepthook = debug

    log.info('creating predictor')
    predictor = MlpPredictor()
    predictor.prepare_usage(args)
    log.info('starting prediction')
    predictions = predictor.run()

    log.info('storing results')
    with utf8_file_open(args.result_file, 'w') as outfile:

        for p in predictions:

            if args.binary:
                outfile.write(unicode((p > 0.5).astype(int)) + u'\n')
            else:
                outfile.write(unicode(p) + u'\n')

    log.info('finished')

if __name__ == '__main__':
    sys.exit(main())
