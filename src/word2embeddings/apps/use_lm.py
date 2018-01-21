# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
"""

from argparse import ArgumentParser
from logging import getLogger
import logging
import sys

from cis.deep.utils import logger_config
from word2embeddings.nn.predictor import vLblNCEPredictor
from word2embeddings.tools.util import debug


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser()
parser.add_argument('predict_file',
        help='Document with examples to predict the label of.')

parser.add_argument('result_file',
        help='Document to which the predictions will be written.')

parser.add_argument('vocabulary',
        help='Vocabulary file that contains list of tokens.')

parser.add_argument('load_model',
        help='Proceed training with the given model file.')


parser.add_argument('--predict-vocabulary', dest='pred_vocab',
        help='Vocabulary that contains the items that should be considered ' +
        'during perplexity computation.\n' +
        'Caution: Make sure this includes <UNK>.\n' +
        'Caution2: If this vocabulary does not contain a word that is seen ' +
        'in prediction this word is not considered during perplexity  ' +
        'calculation.')

parser.add_argument('--batch-size', dest='batch_size', type=int, default=100)


parser.add_argument('-a', '--store_argmax', action='store_true',
        help='Store the most likely vocabulary item.')

parser.add_argument('-r', '--store_rank', action='store_true',
        help='Store the rank of each vocabulary entry according to the ' +
        'softmax.')

parser.add_argument('-sm', '--store_softmax', action='store_true',
        help='Store the whole softmax distributions. Caution: The vocabulary ' +
        'size can be very high. Therefore, the softmax output, which is a ' +
        'distribution over all vocabulary items, might become very large, too.')

parser.add_argument('-nr', '--normalize_with_root', action='store_true',
        help='Compute the root of the sm distribution and normalize the ' +
        'vectors to unit length. This only has an effect when -sm is given.')

parser.add_argument('-ppl', '--perplexity', action='store_true',
        help='Instead of calculating only the other model outputs, e.g., ' +
        'softmax, etc., also compute the perplexity on the given text. ' +
        'If this parameter is given, the predict_file parameter must point ' +
        'to a text file that is iterate over just as in the training, i.e., ' +
        'using a window approach. That means, it does not handle single ' +
        'contexts per line anymore. Caution: does only work for vLBL models. ' +
        'Note: using ppl in validation is slower.')

parser.add_argument('-save-word', '--save_word', action='store_true',
        help='Works only with -ppl parameter. Store next to probability word ' +
        'from prediction vocabulary. Used during post-processing for the '+
        'right interpolation.')

parser.add_argument('-pr', '--predictions', action='store_true',
        help='Store predicted embeddings for each context.')

parser.add_argument('-i', '--information', action='store_true',
        help='Store additional information for every prediction, e.g., (k ' +
        'nearest neighboring words).')

parser.add_argument('--debug-host', dest='debug_host',
        help='Allow remote debugging at the given host IP. Make sure you ' +
        'follow the instructions at ' +
        'http://pydev.org/manual_adv_remote_debugger.html. Especially, the ' +
        'pydevd source must be in the PYTHONPATH and ' +
        'PATHS_FROM_ECLIPSE_TO_PYTHON in pydevd_file_utils.py must be adapted.')


def main(argv=None):
    log.info('started application')

    log.warning('This script is obsolete. It will not be updated anymore and ' +
        'will be deleted in the future. Use use_model.py instead.')

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args()
    log.info('start parameters: ' + str(args))

    if args.debug_host:
        import pydevd
        pydevd.settrace(host=args.debug_host, stdoutToServer=True,
                stderrToServer=True)

    if log.level == logging.DEBUG:
        sys.excepthook = debug

    log.info('creating predictor')
    predictor = vLblNCEPredictor()
    predictor.prepare_usage(args)
    log.info('starting prediction')
    predictor.run()
    log.info('finished')

if __name__ == '__main__':
    sys.exit(main())
