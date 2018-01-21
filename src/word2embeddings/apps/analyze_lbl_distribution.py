# -*- coding: utf-8 -*-
"""
"""
from argparse import ArgumentParser
from logging import getLogger
import sys

from cis.deep.utils import logger_config, embeddings, file_line_generator, \
    utf8_file_open
import numpy as np


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(description='Analyze the most likely tokens given a ' +
        'context and their probabilities.')
parser.add_argument('vocabulary', help='vocabulary')
parser.add_argument('distributions',
        help='file containing the LBL predictions')
parser.add_argument('contexts',
        help='file containing contexts')
parser.add_argument('out_file',
        help='result file')

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')
    vocab = embeddings.read_vocabulary_file(args.vocabulary, False)
    contexts = list(file_line_generator(args.contexts))
    dists = np.loadtxt(args.distributions)

    log.info('computing results')
    # Add X in the n-grams' centers
    # Assume we have the same context size left and right.
    x_pos = len(contexts[0].split()) // 2
    contexts = [sp[:x_pos] + ['X'] + sp[x_pos:]
            for sp in [c.split() for c in contexts]]

    # Sorts all words for each context descending.
    sort_words_per_context_value = np.sort(dists, 1)[: , ::-1]
    sort_words_per_context_idx = np.argsort(dists, 1)[: , ::-1]

    # Sorts all contexts according to their probability assigned to "similar".
    sort_context_for_similar_idx = np.argsort(dists[:, 465])[::-1]
    sort_context_for_similar_value = np.sort(dists[:, 465])[::-1]

    log.info('writing data data')

    with utf8_file_open(args.out_file, 'w') as likelihood_file:

        # Write results to a file
        for (i, idx) in enumerate(sort_context_for_similar_idx):
            likelihood_file.write(u' '.join(contexts[idx]) + u'\t' +
                    unicode(sort_context_for_similar_value[i]) + u'\n')

            # 10 most likely words for the current context
            for j in xrange(10):
                likelihood_file.write(vocab[sort_words_per_context_idx[idx, j]] +
                        u'\t' + unicode(sort_words_per_context_value[idx, j]) +
                        u'\n')

            likelihood_file.write(u'\n')

    log.info('finished')

if __name__ == '__main__':
    sys.exit(main())
