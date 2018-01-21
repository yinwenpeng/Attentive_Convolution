# -*- coding: utf-8 -*-
"""
"""

from argparse import ArgumentParser
from logging import getLogger
import sys

from cis.deep.utils import file_line_generator, logger_config, utf8_file_open,\
    log_iterations


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Takes two files and combines each line in file 1 with
        all lines in file 2.""")
parser.add_argument('file1')
parser.add_argument('file2',
        help="""use the smaller file as file2, it will be kept in memory""")
parser.add_argument('out_file',
        help="""File to write the combination of both files into.
        Bz2 is supported.""")
parser.add_argument('-s', '--separator', default=u' ')

def main(argv=None):
    """See argument parser description."""

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')
    file2_content = list(file_line_generator(args.file2))

    log.info('combining files')

    with utf8_file_open(args.out_file, 'w') as outfile:

        for c, line1 in enumerate(file_line_generator(args.file1)):
            log_iterations(log, c, 1000)

            for line2 in file2_content:
                outfile.write(line1 + args.separator + line2 + u'\n')

    log.info('finished')


if __name__ == "__main__":
    sys.exit(main())
