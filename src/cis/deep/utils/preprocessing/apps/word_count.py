# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from collections import Counter
from logging import getLogger
import sys

from cis.deep.utils import utf8_file_open, logger_config, sort_dict_by_label


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Count all tokens in the given input file and writes them
        with its count to the output file in descending order.""")
parser.add_argument('-l', '--lowercase', action='store_true',
        help='lowercase words before counting')
parser.add_argument('infile', help='input file')
parser.add_argument('outfile', help='output file')

def main(argv=None):
    """See argument parser description."""

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    counter = Counter()

    with utf8_file_open(args.infile, 'r') as infile:

        for line in infile:
            line = line.strip()

            if args.lowercase:
                line = line.lower()
#             line = line.decode('utf-8').strip()

#             log.info(line)
#             if line == '' or line.startswith('<doc id='):
#                 continue

            counter.update(line.strip().split())

    with utf8_file_open(args.outfile, 'w') as outfile:
        for (key, count) in sort_dict_by_label(counter, True):
            outfile.write(u'%s\t%i\n' % (key, count))

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
