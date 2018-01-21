# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
Example usage:
--dev-file X:\sa\experiments\contextual_polarity\mlp\easy_test\1-1-features-test
--epochs-limit 10
--batch-size 2
--examples-limit 4
--dump-period -1
--validation-period 1000
--error-function "cross_entropy"
X:\sa\experiments\contextual_polarity\mlp\easy_test\1-1-features
1
1
"1"
"""

from argparse import ArgumentParser
from logging import getLogger
import logging
import sys

from cis.deep.utils import logger_config
from word2embeddings.nn.trainer import MlpTrainer
from word2embeddings.tools.util import debug


# import cProfile
# CAUTION: remove the Theano path before importing any of my or Theanos
# libraries 
# print '\n'
if 'C:\\Anaconda\\lib\\site-packages\\theano-current' in sys.path:
    sys.path.remove('C:\\Anaconda\\lib\\site-packages\\theano-current')
    print 'removed old theano path'
# Calculus
if '/usr/local/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg' in sys.path:
    sys.path.remove('/usr/local/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg')
    print 'removed old theano path'
# Omega
# if '/usr/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg' in sys.path:
#     sys.path.remove('/usr/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg')
#     print 'removed old theano path'
# print '\n'.join(sys.path)
# print '\n'
# exit()


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser()
parser.add_argument('train_file',
        help='Document for training that contains tokenized text')
parser.add_argument('input_size', type=int, help='size of the input')
parser.add_argument('output_size', type=int, help='size of the output')
parser.add_argument('hidden_layers', default='32',
        help='Width of each hidden layer, comma separated. E.g., "128,64,32"')

parser.add_argument('--dev-file', dest='dev_file',
        help='Document for dev that contains tokenized text. If no file ' +
        'is given validation will only be performed on the training data.')

parser.add_argument('--disable-padding', dest='disable_padding',
        action='store_true', default=False,
        help='Disable padding sentences while generating examples')

parser.add_argument('--load-model', dest='load_model',
        help='Proceed training with the given model file.')

# Argument for MiniBatchTrainer
parser.add_argument('--epochs-limit', dest='epochs_limit',
        type=int, default=1)
parser.add_argument('--batch-size', dest='batch_size', type=int, default=16)
parser.add_argument('--learning-rate', dest='learning_rate',
        type=float, default=0.1)
parser.add_argument('--decay-learning', dest='decay_learning',
        choices=['linear'], default='', help='Supports "linear" decay for now.')
parser.add_argument('--learning-method', dest='learning_method',
        choices=['fan_in', 'global'], default='global',
        help='Determine the method that learning rate is calculated. Two ' +
        'options are available: {fan_in, global}')
parser.add_argument('--dump-period', dest='dump_period', type=int,
        default=1800,
        help='A model will be dumped every x seconds (-1 for never, i.e., ' +
        'only the final and the best model after training will be dumped.)')
parser.add_argument('--validation-period', dest='validation_period',
        type=float, default=5e5,
        help='A model will be evaluated every y seconds/examples. (-1 ' +
        'for never). If a development file is given, the scores on the ' +
        'training data and the validation data is computed, otherwise only ' +
        'the former is computed.')
parser.add_argument('--period-type', dest='period_type', default='examples',
        choices=['time', 'examples'],
        help='Set the period to be in seconds or number of examples ' +
        'by setting the option to time or examples.')
parser.add_argument('--save-best', dest='save_best', action='store_true',
        help='Save the best model every validation period.')
parser.add_argument('--dump-each-epoch', dest='dump_each_epoch',
        action='store_true', help='Dump the model after each epoch')
parser.add_argument('--examples-limit', dest='examples_limit', type=float,
        help='Size of example to be used', default=1e9)
parser.add_argument('--error-function', dest='error_func',
        default='least_squares', choices=['cross_entropy', 'least_squares'],
        help='defines the used error function (default: least_squares)')

def main(argv=None):
    log.info('started application')

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args()
    log.info('start parameters: ' + str(args))

    if log.level == logging.DEBUG:
        sys.excepthook = debug

    log.info('creating trainer')
    trainer = MlpTrainer()
    trainer.prepare_usage(args)
    log.info('starting training')
    trainer.run()
    log.info('finished')

if __name__ == '__main__':
#     cProfile.run('main()')
    
    sys.exit(main())
