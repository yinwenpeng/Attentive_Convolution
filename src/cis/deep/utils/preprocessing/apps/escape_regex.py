# -*- coding: utf-8 -*-
"""
example usage:
"""

from argparse import ArgumentParser
from logging import getLogger
import re
import sys

from cis.deep.utils import utf8_file_open, logger_config


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(description="""Escape the given text file to remove all
        regular expressions.""")
parser.add_argument('infile',
        help='file that might contain regular expressions')
parser.add_argument('outfile', help='file having regular expressions escaped')

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('transforming data')

    with utf8_file_open(args.infile) as infile:
        with utf8_file_open(args.outfile, 'w') as outfile:

            for line in infile:
                outfile.write(re.escape(line))
    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
