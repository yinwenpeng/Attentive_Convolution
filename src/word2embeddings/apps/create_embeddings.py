# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
"""
from argparse import ArgumentParser
from logging import getLogger
import logging
import sys

# from word2embeddings.apps import use_theano_development_version
# use_theano_development_version()

from cis.deep.utils import logger_config
from word2embeddings.nn.trainer import HingeSentimentMiniBatchTrainer, \
    HingeSentiment2MiniBatchTrainer, HingeMiniBatchTrainer, \
    SimpleVLblNceTrainer, SimpleVLblNceSentimentTrainer, \
    VLblNceTrainer, VLblNceSentimentTrainer, VLblNceDistributionalTrainer, \
    NlblNceTrainer, NvLblNceTrainer, SLmNceTrainer, LblNceTrainer
from word2embeddings.tools.util import debug

log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser()
parser.add_argument('train_file',
        help='Document for training that contains tokenized text')

parser.add_argument('--hidden-layers', dest='hidden_layers',
        help='Width of each hidden layer, comma separated. E.g., ' +
        '"28,64,32". This option only has an effect for mlp models and ' +
        'for slm, where only one hidden layer is allowed.')

parser.add_argument('vocabulary',
        help='Vocabulary file that contains list of tokens.\nCaution: Add ' +
        'the special tokens <UNK>, <S>, </S>, <PAD> in this exact order at ' +
        'the first positions in the vocabulary.')


parser.add_argument('--sentiment-vocabulary', dest='sent_vocab',
        help='Vocabulary file that contains sentiment words')

parser.add_argument('--predict-vocabulary', dest='pred_vocab',
        help='Vocabulary that contains the items that should be considered ' +
        'during perplexity computation.\n' +
        'Caution: Make sure this includes <UNK>.\n' +
        'Caution2: If this vocabulary does not contain a word that is seen ' +
        'in prediction this word is not considered during perplexity  ' +
        'calculation.')


parser.add_argument('--unigram', dest='unigram',
        help='file containing the unigram count (the probabilities are ' +
        'calculated automatically given the counts\n ' +
        'Caution: Add the ' +
        'special tokens <UNK>, <S>, </S>, <PAD> in this exact order at the ' +
        'first positions in the vocabulary.')
parser.add_argument('--noise-samples', dest='noise_samples', type=int,
        help='number of noise samples per data sample')
parser.add_argument('--nce-seed', dest='nce_seed', type=int, default=2345,
        help='seed for the noise sample generation in NCE')


parser.add_argument('--validation-file', dest='validation_file', nargs='+',
        help='Files for validation that contains tokenized text. Multiple ' +
        'files are supported, with the first file being the main validation ' +
        'file, i.e., if --dump-best is active, then the performance on the ' +
        'first file is considered.\n ' +
        'Note: For all LBL based models the validation cost will be ' +
        'different even if you provide the same validation file twice, ' +
        'because the NCE cost computation involves a randomized process.')

parser.add_argument('--perplexity', action='store_true',
        help='instead of calculating the error on the validation set, ' +
        'additionally calculate the perplexity. Caution: does only work ' +
        'for vLBL models. Note: using ppl in validation is slower.')


parser.add_argument('--disable-padding', dest='disable_padding',
        action='store_true', default=False,
        help='Disable padding sentences while generating examples')

parser.add_argument('--learn-eos', dest='learn_eos',
        action='store_true', default=False,
        help='Learn word embedding for the end-of-sentence token </S>.')


parser.add_argument('--load-model', dest='load_model',
        help='Proceed training with the given model file.')

parser.add_argument('--model-type', dest='model_type',
        choices=['ColWes08', 'sent_1', 'sent_2', 'vlbl', 'nvlbl',
                'vlbl_sent', 'simple_vlbl', 'simple_vlbl_sent', 'vlbl_dist',
                'lbl', 'nlbl', 'slm'],
                default='ColWes08',
        help='Type of the model to use for training. All sentiment models ' +
        'require a sentiment vocabulary.')

parser.add_argument('--activation-func', dest='activation_func', default='rect',
        choices=['sigmoid', 'tanh', 'rect', 'softsign'],
        help='Activation function to use in non-linear models.')


parser.add_argument('--left-context', dest='left_context', type=int,
        default=2,
        help='Left context window to be used measured from the current token')

parser.add_argument('--right-context', dest='right_context', type=int,
        default=2,
        help='Right context window measured from the current token')

parser.add_argument('--word-embedding-size', dest='word_embedding_size',
        type=int, default=64)


# Argument for MiniBatchTrainer
parser.add_argument('--epochs-limit', dest='epochs_limit', type=int, default=-1,
        help='maximal number of epochs to train (-1 for no limit)')

parser.add_argument('--examples-limit', dest='examples_limit', type=int,
        default=-1,
        help='maximal number of examples to train (-1 for no limit)')

parser.add_argument('--early-stopping', dest='early_stopping', type=int,
        default=-1,
        help='Stop the training when N consecutive validations resulted in ' + \
        'worse results than the validation before. -1 to deactivate this ' + \
        'feature.')


parser.add_argument('--batch-size', dest='batch_size', type=int, default=16)


parser.add_argument('--learning-rate', dest='learning_rate',
        default=0.1,
        help='Learning rate. If this parameter is a float value than the ' +
        'learning rate is valid for all model parameters. Otherwise, it can ' +
        'contain parameter specific learning rates in using the pattern ' +
        '"param_name1:param_learning_rate1,param_name2:param_learning_rate2\.' +
        'You can also specify a learning rate for only some of your ' +
        'parameters and assign the default learning rate for all other ' +
        'parameters by specifying "default:default_learning_rate".')

parser.add_argument('--lr-adaptation', dest='lr_adaptation_method',
        choices=['constant', 'linear', 'adagrad', 'MniTeh12'],
        default='constant',
        help='Sets the method that is used to reduce the learning rate. ' +
        'Supports "linear" (linear reduction) and "adagrad" (AdaGrad ' +
        'algorithm), and "constant" (no reduction), "MniTeh12" (halves the  ' +
        'learning rate whenever the validation perplexity (if "--perplexity" ' +
        'is given) or error (otherwise) goes up; for details see [MniTeh12])')

parser.add_argument('--learning-method', dest='learning_method',
        choices=['fan_in', 'global'], default='global',
        help='Determine the method that learning rate is calculated. Two ' +
        'options are available: {fan_in, global}')


parser.add_argument('--l1-weight', dest='l1_weight', type=float, default=0.0,
        help='Weight of L1 regularization term. 0 to deactivate. ' +
        'Only implemented for LBL models and SLM.')
parser.add_argument('--l2-weight', dest='l2_weight', type=float, default=0.0,
        help='Weight of L2 regularization term. 0 to deactivate. ' +
        'Only implemented for LBL models and SLM.')

parser.add_argument('--dump-period', dest='dump_period', type=int, default=-1,
        help='A model will be dumped every x seconds/examples (-1 = no ' +
        'dumping. Only the final model will be dumped.)')

parser.add_argument('--load-params', dest='load_params', nargs=2,
        help='Load initial values from files. This parameter requires two ' +
        'arguments: (i) <BASE_FILENAME> and (ii) a comma separated list of ' +
        'parameter names as specified by the individual model. Each parameter' +
        'must be stored in csv file format in an own file. The single ' +
        'parameter files are then expected to be named ' +
        '<BASE_FILENAME>.<PARAMETER_NAMES>.\n ' +
        'Example usage: ~/my_model "C,R" will load ~/my_model.C and ' +
        '~/my_model.R.\n ' +
        'Gzip and bz2 files are supported.')

parser.add_argument('--store-params', dest='store_params',
        help='Comma-separated list of parameter names that will be stored ' +
        'each time the model is stored. The parameter names as specified by ' +
        'the individual model. Each parameter is stored in a separate file, ' +
        'e.g., paramter C is stored in <MODEL_NAME>.params.C.')

parser.add_argument('--out-dir', dest='out_dir', default='.',
        help='directory where to store the output files')

parser.add_argument('--dump-vocabulary', dest='dump_vocabulary',
        action='store_true',
        help='Dump the vocabulary after importing it to remove duplicates.')

parser.add_argument('--dump-embeddings', dest='dump_embeddings',
        action='store_true',
        help='Dump the embeddings for every dumped model. Caution: might ' +
        'be a big file.\n ' +
        'Caution: This parameter is deprecated. It\'s not supported by the ' +
        'new vLBL models. Use --store-params instead.')

parser.add_argument('--validation-period', dest='validation_period',
        type=float, default=-1,
        help='A model will be evaluated every y seconds/examples. (-1 ' +
        'for never). If a development file is given, the scores on the ' +
        'training data and the validation data is computed, otherwise only ' +
        'the former is computed.')

parser.add_argument('--period-type', dest='period_type', default='examples',
        choices=['time', 'examples'],
        help='Set the period to be in seconds or number of examples ' +
        'by setting the option to time or examples.')

parser.add_argument('--dump-best', dest='dump_best', action='store_true',
        help='Save the best model every validation period. What "best" ' + \
        'means depends on the type of model. If "--perplexity" is given, ' + \
        'it\'s the model with the lowest perplexity. If not, it\'s the ' + \
        'model with the lowest training error.')

parser.add_argument('--dump-each-epoch', dest='dump_each_epoch',
        action='store_true', help='Dump the model after each epoch')

parser.add_argument('--dump-initial-model', dest='dump_initial_model',
        action='store_true',
        help='Dump the initial model before any training is done.')


parser.add_argument('--error-function', dest='error_func',
        default='least_squares', choices=['cross_entropy', 'least_squares'],
        help='defines the used error function (default: least_squares); ' +
        'This parameter is only valid for MLPs.')

parser.add_argument('--count-examples', dest='count_examples',
        action='store_true',
        help='Only count the examples in the training file, don\'t train a ' +
        'model.')


parser.add_argument('--debug-host', dest='debug_host',
        help='Allow remote debugging at the given host IP. Make sure you ' +
        'follow the instructions at ' +
        'http://pydev.org/manual_adv_remote_debugger.html. Especially, the ' +
        'pydevd source must be in the PYTHONPATH and ' +
        'PATHS_FROM_ECLIPSE_TO_PYTHON in pydevd_file_utils.py must be adapted.')

def main(argv=None):
    log.info('started application')

    log.warning('This script is obsolete. It will not be updated anymore and ' +
        'will be deleted in the future. Use train_model.py instead.')

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

    check_args(args)

    log.info('start parameters: ' + str(args))

    if args.debug_host:
        import pydevd
        pydevd.settrace(host=args.debug_host, stdoutToServer=True,
                stderrToServer=True)

    if log.level == logging.DEBUG:
        sys.excepthook = debug

    log.info('creating trainer')

    if args.model_type == 'ColWes08':
        log.info('Using ColWes08 trainer')
        trainer = HingeMiniBatchTrainer()
    elif args.model_type == 'sent_1':
        log.info('Using sent_1 trainer')
        trainer = HingeSentimentMiniBatchTrainer()
    elif args.model_type == 'sent_2':
        log.info('Using sent_2 trainer')
        trainer = HingeSentiment2MiniBatchTrainer()
    elif args.model_type == 'simple_vlbl':
        log.info('Using simple LBL trainer that uses noise-contrastive estimation')
        trainer = SimpleVLblNceTrainer()
    elif args.model_type == 'simple_vlbl_sent':
        log.info('Using simple LBL trainer that uses noise-contrastive estimation to create sentiment embeddings')
        trainer = SimpleVLblNceSentimentTrainer()
    elif args.model_type == 'vlbl':
        log.info('Using LBL trainer that uses noise-contrastive estimation')
        trainer = VLblNceTrainer()
    elif args.model_type == 'vlbl_sent':
        log.info('Using LBL trainer that uses noise-contrastive estimation to create sentiment embeddings')
        trainer = VLblNceSentimentTrainer()
    elif args.model_type == 'nvlbl':
        log.info('Using non-linear vLBL NCE trainer')
        trainer = NvLblNceTrainer()
    elif args.model_type == 'lbl':
        log.info('Using linear LBL trainer that uses noise-contrastive estimation')
        trainer = LblNceTrainer()
    elif args.model_type == 'nlbl':
        log.info('Using non-linear LBL trainer that uses noise-contrastive estimation')
        trainer = NlblNceTrainer()
    elif args.model_type == 'vlbl_dist':
        log.info('Using LBL trainer that uses distributional representation of input')
        trainer = VLblNceDistributionalTrainer()
    elif args.model_type == 'slm':
        log.info('Using shallow neural network lm with NCE')
        trainer = SLmNceTrainer()
    else:
        raise ValueError('Unknown model type. Abort')

    if args.count_examples is True:
        log.info('counting examples')
        trainer.configure(args)
        count = trainer.count_examples(args.train_file)
        log.info('examples: %d' % count)
    else:
        trainer.prepare_usage(args)
        log.info('training is about to begin')
        trainer.run()

    log.info('finished')

def check_args(args):



#     if args.epochs_limit == -1 and args.examples_limit == -1:
#         raise ValueError('Either epochs-limit or examples-limit must be given.')
    pass

if __name__ == '__main__':
    sys.exit(main())
