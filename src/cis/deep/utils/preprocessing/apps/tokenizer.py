# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from logging import getLogger
import sys

from cis.deep.utils import logger_config, utf8_file_open
from cis.deep.utils.text import tokenize


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Tokenizes the given input file by NLTK\'s recommended
        word tokenizer and writes the result into the output file.""")
parser.add_argument('infile', help='input file')
parser.add_argument('outfile', help='output file')

def main(argv=None):
    """See argument parser description."""

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    with utf8_file_open(args.infile, 'r') as infile:
        with utf8_file_open(args.outfile, 'w') as outfile:

            for line in infile:
                outfile.write(' '.join(tokenize(line)) + '\n')

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
