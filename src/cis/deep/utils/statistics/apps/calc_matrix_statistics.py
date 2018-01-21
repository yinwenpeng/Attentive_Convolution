# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from logging import getLogger
import sys

from cis.deep.utils import utf8_file_open, logger_config, log_iterations
from cis.deep.utils.statistics import calc_matrix_statistics


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Calculates the basic statistics for a file that contains
        a matrix in csv format with spaces as separators. The statistics include
        mean, max, min, and std dev for every row in the input file.""")
parser.add_argument('infile', type=str, help='input file')
parser.add_argument('outfile', type=str, help='output file')

def main(argv=None):
    """See argument parser description."""

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    with utf8_file_open(args.outfile, 'w') as outfile:
        outfile.write(u'mean max min std_dev\n')

        for (count, tupel) in enumerate(calc_matrix_statistics(args.infile)):
            log_iterations(log, count, 10000)

            outfile.write(u'%f %f %f %f\n' % tupel)

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
