# -*- coding: utf-8 -*-
"""
X:\sa\experiments\contextual_polarity\vlbl\sentiment-wnd3_3-nce5/classification\ebert,20140515-n_grams
X:\sa\embeddings\vlbl\sentiment-wnd3_3-nce5\vLBL.vocab
./embs.txt
./features_out
"""

from argparse import ArgumentParser
from logging import getLogger
import os
import sys

from cis.deep.utils import file_line_generator, logger_config, utf8_file_open, \
    ndarray_to_string
from cis.deep.utils.embeddings import read_vocabulary_id_file, SpecialTokenID
import numpy as np


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Converts a given text file into a features file. This
        is done by replacing each token in the text file by it's given feature
        vector. All features will be concatenated, i.e., there will be no space
        between.""")
parser.add_argument('infile',
        help="""Data file, containing all tokens to be replaced by their
        features. The file can be compressed with bz2 or gz.""")
parser.add_argument('vocabulary',
        help="""Vocabulary file containing all valid words. Tokens not contained
        in the vocabulary will be mapped to <UNK>.""")
parser.add_argument('feature_file',
        help="""File containing all token features. Each feature must be in a
        single row. The row index must correspond to the vocabulary index.
        Currently, only dense matrices are supported.""")
parser.add_argument('out_feature_file',
        help="""File to write the features into. Bz2 is supported.""")

parser.add_argument('-a', '--avg', action='store_true',
        help='Average the features for all words in one example (i.e., line).')


def main(argv=None):
    """See argument parser description."""

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')
    vocab = read_vocabulary_id_file(args.vocabulary, False)

    _, ext = os.path.splitext(args.feature_file)

    if ext == 'npy':
        features = np.load(args.feature_file)
    else:
        features = np.loadtxt(args.feature_file)

    log.info('creating features')

    with utf8_file_open(args.out_feature_file, 'w') as outfile:

        for line in file_line_generator(args.infile):
            toks = line.split()
            cur_features = np.zeros((len(toks), features.shape[1]))

            for (i, tok) in enumerate(toks):
                cur_features[i, :] = features[
                        vocab.get(tok, SpecialTokenID.UNKNOWN.value)]

            if args.avg:
                res = ndarray_to_string(np.mean(cur_features, axis=0))
            else:
                res = ndarray_to_string(np.reshape(cur_features,
                        np.prod(cur_features.shape), order='C'))

            outfile.write(res + u'\n')

    log.info('finished')


if __name__ == "__main__":
    sys.exit(main())
