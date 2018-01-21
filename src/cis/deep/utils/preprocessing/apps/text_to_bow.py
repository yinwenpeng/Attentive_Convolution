# -*- coding: utf-8 -*-
"""
-v X:\sa\embeddings\vlbl\sentiment-wnd3_3-nce5\vlbl.vocab
ebert,20140515-n_grams
ebert,20140515-n_grams.out
"""

from argparse import ArgumentParser
from logging import getLogger
import sys

from sklearn.feature_extraction.text import CountVectorizer

from cis.deep.utils import file_line_generator, logger_config, utf8_file_open
from cis.deep.utils.embeddings import read_vocabulary_id_file
import numpy as np


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Converts a given text file into a bag-of-words feature
        file. Currently, only tf is supported.""")
parser.add_argument('infile',
        help="""Data file, containing all tokens. Each line will get its 
        bow feature vector.""")
parser.add_argument('out_feature_file',
        help="""File to write the features into. Bz2 is supported.""")

parser.add_argument('-v', '--vocabulary',
        help="""Vocabulary file containing all valid words. If it's not given
        the vocabulary is inferred and stored afterwards. For additional
        information see
        sklearn.feature_extraction.text.CountVectorizer.__init__'s vocabulary
        parameter.""")
parser.add_argument('-n', '--ngram', default='1,1',
        help="""comma-separated list of (min n-gram, max n-gram). For example
        "1,3" includes all unigrams, bigrams, and trigrams. For additional
        information see see CountVectorizer.__init__'s ngram_range parameter.
        """)

def main(argv=None):
    """See argument parser description."""

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')

    if args.vocabulary is None:
        vocab = args.vocabulary
    else:
        vocab = read_vocabulary_id_file(args.vocabulary)

    text = list(file_line_generator(args.infile))

    ngram_range = map(int, tuple(args.ngram.split(',')))
    vectorizer = CountVectorizer(token_pattern='[^ ]+', min_df=0.0,
            vocabulary=vocab, ngram_range=ngram_range, dtype=int)

    log.info('creating features')
    bow = vectorizer.fit_transform(text)

    log.info('storing result')
    np.savetxt(args.out_feature_file, bow.todense(), fmt='%d')

    with utf8_file_open(args.out_feature_file + '.vocab', 'w') as vocab_file:
        vocab_file.write(u'\n'.join(vectorizer.get_feature_names()))

    log.info('finished')


if __name__ == "__main__":
    sys.exit(main())
