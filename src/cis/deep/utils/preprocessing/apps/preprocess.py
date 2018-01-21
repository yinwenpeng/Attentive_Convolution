# -*- coding: utf-8 -*-
"""
Example usage:
--amazon
NLTK_DATA_DIR = 'C:/Temp/NLTK data'
x
y
"""

from argparse import ArgumentParser
from logging import getLogger
import sys

from cis.deep.utils import logger_config, file_line_generator, utf8_file_open, \
    log_iterations
from cis.deep.utils.preprocessing.corpus import AmazonProductReviewCorpusReader
import re
from cis.deep.utils.text import tokenize
import nltk

log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(description="""
        Preprocess a given file. Several preprocessing parameters are
        available. TODO: add lowercasing""")
parser.add_argument('--amazon', action='store_true',
        help="""preprocess the Amazon product review corpus.""")

parser.add_argument('-rd', '--replace_digits',
        help="""Replace all digits by the given string""")
parser.add_argument('-sh', '--strip_html', action='store_true',
        help='strip html tags')
parser.add_argument('-t', '--tokenize', action='store_true',
        help="""tokenize the text""")
parser.add_argument('-ss', '--sentence_splitter', type=str,
        default='tokenizers/punkt/english.pickle',
        help='model file to be used for sentence splitting (default: ' + \
        'tokenizers/punkt/english.pickle)')
parser.add_argument('-s', '--split_sentence', action='store_true',
        help='split sentences')
parser.add_argument('infile', help='name of the input file')
parser.add_argument('outfile', help='name of the output file')

REGEX_FLAGS = re.UNICODE

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))
    log.info('preprocessing data')

    if args.amazon is True:
        line_iterator = \
                AmazonProductReviewCorpusReader(args.infile).review_generator()
    else:
        line_iterator = file_line_generator(args.infile)

    if args.sentence_splitter:
        sent_splitter = nltk.data.load(args.sentence_splitter)

    with utf8_file_open(args.outfile, 'w') as outfile:

        for (i, line) in enumerate(line_iterator):
            log_iterations(log, i, 100000)

            if args.replace_digits:
                line = re.sub(r'\d', args.replace_digits, line,
                        0, REGEX_FLAGS)

            if args.strip_html:
                line = nltk.clean_html(line)

            if args.sentence_splitter:
                line = sent_splitter.tokenize(line)
            else:
                line = [line]

            if args.tokenize:
                line = [tokenize(l) for l in line]

            if not args.tokenize:
                outfile.write(u'\n'.join(line))
            else:
                outfile.write(u'\n'.join([u' '.join(l) for l in line]))

            outfile.write(u'\n')

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
