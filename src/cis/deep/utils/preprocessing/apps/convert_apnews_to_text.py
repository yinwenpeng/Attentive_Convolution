# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from logging import getLogger
import sys

import numpy as np
import pandas as pd

from cis.deep.utils import utf8_file_open, logger_config, file_line_generator


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Converts the binary files of the AP News (Associated
        News) corpus provided by Yoshua Bengio into readable text.""")
parser.add_argument('infile', type=str, help='input file')
parser.add_argument('outfile', type=str, help='output file')
parser.add_argument('vocabulary', type=str, help='vocabular file')

def main(argv=None):
    """See argument parser description."""

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    vocab = pd.Series(file_line_generator(args.vocabulary, comment='##'))

    with open(args.infile, 'rb') as infile:
        integers = np.fromfile(infile, np.int32)

    with utf8_file_open(args.outfile, 'w') as outfile:
        outfile.write(u'\n'.join(vocab[integers]))
        outfile.write(u'\n')

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
