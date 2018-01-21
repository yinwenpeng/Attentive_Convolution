# -*- coding: utf-8 -*-
"""
This file contains an application that extracts the vocabulary and embeddings
from a given model file.
"""
from argparse import ArgumentParser
from logging import getLogger
import sys

from cis.deep.utils import logger_config, load_object_from_file


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(description='extract parameters from a given model ' +
        'file')
parser.add_argument('model_file', help='model file')
parser.add_argument('store_params', nargs=2,
        help='The first parameter is the filename, the second is a ' +
        'comma-separated list of parameter names. For more information see ' +
        'the documentation of the --load-params parameter in ' +
        'create_embeddings.py.')

parser.add_argument('-f', '--format', default='txt', choices=['txt', 'npy'],
        help='Format of the output files. txt = space separated csv format; ' +
        'npy = binary numpy format')

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')
    model = load_object_from_file(args.model_file)

    log.info('writing data')
#     trainer.dump_vocabulary(args.vocabulary_file)
    model.store_params(args.store_params[0], args.store_params[1], True,
            args.format)
    log.info('finished')

if __name__ == '__main__':
    sys.exit(main())
