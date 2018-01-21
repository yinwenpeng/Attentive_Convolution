# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
"""

from argparse import ArgumentParser
from logging import getLogger
import os
import sys

from cis.deep.utils import logger_config, utf8_file_open, file_line_generator


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Filters a given file by lines indices.""")

parser.add_argument('indices', help="""line numbers that will be included in 
        the output; either comma separated string (e.g., 1,4,6) or file
        containing one index per line;
        Caution: make sure the indices are sorted; the indices are 0-based.""")
parser.add_argument('infile', help='file to be filtered')
parser.add_argument('outfile', help='filtered output file')
parser.add_argument('-i', '--inverse', action='store_true',
        help="""inverse the indices, i.e., exclude the lines with the given
        line number""")

def get_indices(indices):
    """Generates line indices to keep.

    Parameters
    ----------
    indices : str
        either name of a file containing indices one per line or a comma
        separated string

    Returns
    -------
    int
        next index
    """

    if os.path.exists(indices):
        return set(map(int, file_line_generator(indices, True)))

    return set((int(i.strip()) for i in indices.split(u',')))

def main(argv=None):
    log.info('started application')

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args()
    log.info('start parameters: ' + str(args))
    log.info('reading index file')
    idx = get_indices(args.indices)
    max_idx = max(idx)
    log.info('filtering file')

    with utf8_file_open(args.outfile, 'w') as outfile:

        for (cur_idx, line) in enumerate(
                file_line_generator(args.infile, False)):

            if not args.inverse:

                if cur_idx in idx:
                    outfile.write(line)

                if cur_idx >= max_idx:
                    break
            else:

                if cur_idx not in idx:
                    outfile.write(line)


    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
