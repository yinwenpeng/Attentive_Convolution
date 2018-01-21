# -*- coding: utf-8 -*-
"""
This file contains an application that extracts the input and output embeddings
from a given vlbl or vlbl_dist model file.
"""
from argparse import ArgumentParser
from logging import getLogger
import sys
import numpy as np

from cis.deep.utils import logger_config, load_object_from_file, utf8_file_open, sort_dict_by_label
from cis.deep.utils.embeddings import read_vocabulary_id_file

log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(description='Extract input and output word embeddings ' +
        'from a given vLBL or distributional vLBL model file. ' +
        'Output format is "word space embedding". ' +
        'In case of vLBL model, input embeddings are represented with R matrix; ' +
        'output embeddings are represented with Q matrix. ' +
        'In case of vLBL distributional model, input embeddings are ' +
        'represented with D*R matrix, output embeddings are represented with ' +
        'Q matrix.')
parser.add_argument('model_file', help='vlbl or vlbl_dist model file')
parser.add_argument('--model-type', dest='model_type',
        choices=['vlbl', 'vlbl_dist'],
                default='vlbl',
        help='Type of the model to use for embeddings extraction.')
parser.add_argument("vocabulary",
        help="Vocabulary file that contains list of tokens.")
parser.add_argument("result_file",
        help="Document to which the predictions will be written.")

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')
    model = load_object_from_file(args.model_file)

    # read vocabulary from file
    vocab = sort_dict_by_label(read_vocabulary_id_file(args.vocabulary))

    # get matrices from model
    r_matrix = model.R.get_value()
    q_matrix = model.Q.get_value()

    # get input embeddings
    if args.model_type == 'vlbl':
        in_we = r_matrix
    elif args.model_type == 'vlbl_dist':
        # this will not work with the old versions of models - because of sparsity
        d_matrix = model.D.get_value().todense()
        in_we = np.dot(d_matrix, r_matrix)
        # need to convert from numpy.matrix to numpy.ndarray
        in_we = in_we.view(type=np.ndarray)

    with utf8_file_open(args.result_file + ".in", 'w') as outfile:
        for (word, ind) in vocab:
            outfile.write(unicode(word) + u' ' + u' '.join(map(str, in_we[ind])) + u'\n')

    with utf8_file_open(args.result_file + ".out", 'w') as outfile:
        for (word, ind) in vocab:
            outfile.write(unicode(word) + u' ' + u' '.join(map(str, q_matrix[ind])) + u'\n')

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
