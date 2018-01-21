# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from logging import getLogger
import sys

import nltk

from cis.deep.utils import utf8_file_open, logger_config


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Splits the given input file into sentences by NLTK\'s
        punkt sentence tokenizer and writes the result into the output file.
        It assumes English language if there is no language given. It reads
        one line at a time, i.e., if there are line breaks not marking sentence
        boundaries, they won't be handled correctly.""")
parser.add_argument('-m', '--model', type=str,
        default='tokenizers/punkt/english.pickle',
        help='model file to be used for sentence splitting (default: ' + \
        'tokenizers/punkt/english.pickle)')
parser.add_argument('infile', type=str, help='input file')
parser.add_argument('outfile', type=str, help='output file')

def main(argv=None):
    """See argument parser description."""

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    with utf8_file_open(args.infile, 'r') as infile:
        with utf8_file_open(args.outfile, 'w') as outfile:
            sent_splitter = nltk.data.load(args.model)

            for line in infile:
                outfile.write('\n'.join(sent_splitter.tokenize(line.strip())) +
                        '\n')

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
