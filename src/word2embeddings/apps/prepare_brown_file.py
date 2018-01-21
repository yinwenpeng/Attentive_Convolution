# -*- coding: utf-8 -*-
"""
example usage:
-r
18
X:\sa\dictionary\Brown_clusters\brown-mod
X:\sa\dictionary\Brown_clusters\brown-mod-fixed_length_right
"""

from argparse import ArgumentParser
from logging import getLogger
import sys

from cis.deep.utils import utf8_file_open, file_line_generator, logger_config
from word2embeddings.tools.util import prepare_brown_signature


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(description='Prepare a given file that contains ' +
        'Brown clustering signatures for words. Convert the variable length ' +
        'signatures into fixed length signatures.')
parser.add_argument('max_size', help='size of the fixed signatures', type=int)
parser.add_argument('infile', help='input file with variable size signatures')
parser.add_argument('outfile', help='output file with fixed size signatures')
parser.add_argument('-r', '--right', default=False,
        action='store_true',
        help='pad the signatures to the right instead of to the left')

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('transforming data')

    with utf8_file_open(args.outfile, 'w') as outfile:
        for line in file_line_generator(args.infile):
            token, signature = line.split(u'\t')
            outfile.write(u'%s\t%s\n' % (token, prepare_brown_signature(
                    signature, args.max_size, args.right)))

    log.info('finished')

if __name__ == '__main__':
    sys.exit(main())
