#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""trainer.py: Contains multiple training strategies."""

from datetime import datetime
from io import open, BytesIO
import json
from logging import getLogger
import os
from random import randint, choice
from time import time

from theano import config, tensor as T

import cPickle as pickle
from cis.deep.utils import utf8_file_open, load_object_from_file, \
    logger_config, text_to_vocab_indices, log_iterations, generator_has_next
from cis.deep.utils.embeddings import SpecialTokenID, read_vocabulary_id_file, \
    read_vocabulary_file, write_vocabulary_file
from cis.deep.utils.misc import softmax
from cis.deep.utils.theano import debug_print
import numpy as np
from word2embeddings.lm.networks import WordPhraseNetwork
from word2embeddings.nn.layers import floatX, intX, EmbeddingLayer
from word2embeddings.nn.networks import MultiLayerPerceptron, SimpleVLblNce, \
    VLblNce, VLblNceDistributional, NlblNce, NvLblNce, SLmNce, LblNce
from word2embeddings.nn.tools import read_unigram_distribution
from word2embeddings.tools.examples_generator import \
    PaddedWindowExamplesGenerator, SentimentExamplesGenerator, \
    LabeledExampleGenerator, SentimentAnywhereExamplesGenerator
from word2embeddings.tools.util import grouper
from itertools import islice


log = getLogger(__name__)
logger_config(log)

class MiniBatchRunner(object):
    # Specifies the example iterator to be used by this runner.
    example_iterator_type = PaddedWindowExamplesGenerator

    """ Stochastic gradient descent mini-batch trainer."""
    def __init__(self, model=None):
        # The neural network (model) that has to be trained
        self.model = model
        self._datetime_format = '%y-%m-%d_%H-%M-%S'

        # Size of the minibatch
        self.batch_size = 10

    def before_run_begins(self):
        """Is called directly before the training begins."""
        pass

    def _calc_probabilities_from_similarity(self, orig_indices, similarities):
        """Calculate the probability for the given indices according to the
        similarities.

        Parameters
        ----------
        orig_indices : iterable
            indices of the tokens you want the probability for; the indices
            are given from the original vocabulary and will automatically be
            converted to the prediction vocabulary
        similarities : ndarray
            matrix of similarities

        Returns
        -------
        ndarray
            softmax; contains nan values in the rows for which no vocabulary
            mapping was available
        ndarray
            probabilities of the tokens given by orig_indices;
            contains nan values in the rows for which no vocabulary mapping was
            available
        float
            sum of the logarithms of the probabilities;
            is non if no vocabulary mapping was available
        int
            Number of examples the probability could have been computed for.
            This might not be the number of given examples in the case when an
            index from orig_indices does not exist in the vocabulary mapping,
            i.e., the word does not exist in the prediction vocabulary.
        """
        log_probabs = 0.
        num_ppl_examples = 0

        # Get the indices of the rows we have a mapping of indices for, i.e.,
        # discard all input similarities where we don't have a mapping.
        row_to_pred_idx = [(i, self.vocab_mapping[tok])
                for i, tok in enumerate(orig_indices)
                if tok in self.vocab_mapping]

        sm = np.empty((similarities.shape[0], len(self.vocab_mapping_list)))
        sm.fill(np.nan)
        probabs = np.empty((similarities.shape[0],))
        probabs.fill(np.nan)

        # If we have at least 1 valid mapping
        if row_to_pred_idx:
            rows, pred_indices = map(list, zip(*row_to_pred_idx))
            similarities = similarities[:, self.vocab_mapping_list]
            sm[rows, :] = softmax(similarities[rows])
            probabs[rows] = sm[rows, pred_indices]

            # Set 0 probabilities to a very small number. That prevents inf
            # perplexities.
            probabs[probabs == 0] = np.exp(-99)
            log_probabs += np.sum(np.log(probabs[rows]))
            num_ppl_examples += len(rows)
        else:
            log_probabs = np.nan

        return sm, probabs, log_probabs, num_ppl_examples

    def configure(self, args):
        self.batch_size = args.batch_size
        self.load_model = args.load_model

    def count_examples(self, input_file):
        examples = self.prepare_example_generator().example_iter(input_file)
        return sum(1 for _ in examples)

    def epoch_finished(self):
        """
        Handle things to be done after one epoch has finished, e.g., log.
        """
        pass

    def epoch_iter(self, input_file):
        """
        Start a new epoch using the input data from the given file and iterate
        over all examples in the file.

        Parameters
        ----------
        input_file : str
            name of the example file

        Returns
        -------
        any
            returns the processed example
        """
        examples = self.prepare_example_generator().example_iter(input_file)

        for example in examples:
            yield self.process_example(example)

        log.debug('Finished another epoch over all examples')

    def prepare_example_generator(self):
        """Create a new example generator and configure it accordingly.

        Set the class variable example_iterator_type to specify the example
        iterator to be used.
        """
        generator = self.example_iterator_type()
        generator.configure(self)
        return generator;

    def get_name(self):
        date_time = datetime.fromtimestamp(time())
        time_string = date_time.strftime(self._datetime_format)
        return '%s_%d_%s' % (self.name, self.model.total_epochs, time_string)

    def get_model(self):
        """
        Load the network or create a new one depending on the user settings.
        """
        raise NotImplementedError

    @property
    def name(self):
        return self.model.name

    def next_batch(self, data_file):
        """Accumulates examples into batches.

        Parameters
        ----------
        data_file : str
            name of the file to iterate over

        Returns
        -------
        list
            batch of examples
        bool
            Indicator whether or not an epoch was finished during this batch. If
            this is true, the batch might be empty or not entirely filled.
        """
        batch = []
        gen = self.epoch_iter(data_file)
        next_example = gen.next()

        # We cannot use a for loop here, because we immediately need to know if
        # we reached the end of a data file. So we need to perform a "forward
        # lookup" and check if there is one more element in the generator left.
        # If not, an exception is risen and we return from here.
        while True:
            example = next_example
            batch.append(example)

            # Perform the forward lookup, i.e., check if there is an element
            # left in the generator.
            try:
                next_example = gen.next()
            except StopIteration:
                yield batch, True
                return

            if len(batch) == self.batch_size:
                yield batch, False
                batch = []

    def prepare_arguments(self, batch):
        arguments = zip(*batch)
        return arguments

    def prepare_usage(self, options):
        """Configure the trainer, load the network and prepare the usage."""
        self.configure(options)
        self.get_model()

    def run(self):
        raise NotImplementedError()


class MiniBatchTrainer(MiniBatchRunner):

    def __init__(self):
        super(MiniBatchTrainer, self).__init__()
        self.train_monitor = {'costs': 0.0, 'batches': 0}
        self.dev_monitor = {'costs': 0.0, 'batches': 0}

        # Indicates how many examples to skip to continue training where a
        # loaded model stopped. If no skipping is necessary, this variable is
        # -1.
        self.skip_examples = -1
        self.train_total_batches = 0
        self.dev_total_batches = 0
        self.dev_total_costs = 0.0

        self.previous_validation_avg_cost = 1e20
        self.previous_validation_ppl = 1e20

        self.train_total_examples = []
        self.dev_avg_cost_series = []
        self.train_avg_cost_series = []

        self.best_model = None
        self.best_score = None
        self.best_name = None

        # Period in seconds
        self.dump_period = -1
        self.dump_each_epoch = False
        self.validation_period = 120

        # Time in seconds
        self._last_validation = time()
        self._last_dump = time()

        self.time_series = []

    def before_run_begins(self):
        super(MiniBatchTrainer, self).before_run_begins()

        if self.dump_initial_model:
            self.dump(self.model)

    def dump_best_model(self, score, is_better_fn=lambda new, old: new < old):
        """ Saves the model in memory if its better than the existing one"""

        if self.best_model == None or is_better_fn(score, self.best_score):

            if self.best_score == None:
                log.info('First validation phase complete. New score: %f',
                        score)
            else:
                log.info('Saving new best model from validation phase. '
                        'Old score: %f, New score: %f', self.best_score, score)

            start_time = time()

            byteout = BytesIO()
            self.dump(self.model, outstream=byteout)
            self.best_model = byteout
            self.best_score = score
            self.best_name = self.get_name()

            log.info('Best model saved, in time %f s', time() - start_time)

    def configure(self, args):
        super(MiniBatchTrainer, self).configure(args)
        self.disable_padding = args.disable_padding
        self.learn_eos = args.learn_eos
        self.train_file = args.train_file
        self.validation_file = args.validation_file
        self.early_stopping = args.early_stopping

        # Number of consecutive validations that were worse than the one before.
        # Necessary for early stopping.
        self._early_stopping_count = 0
        self.perplexity = args.perplexity
        self.dump_best = args.dump_best
        self.dump_each_epoch = args.dump_each_epoch
        self.dump_initial_model = args.dump_initial_model
        self.period_type = args.period_type
        self.set_period_type(args.period_type)
        self.dump_period = int(args.dump_period)
        self.validation_period = int(args.validation_period)

        try:
            f = float(args.learning_rate)
            self.learning_rate = {'default': f}
        except ValueError:
            # Convert string representation into dict. Representation is
            # "param1:lr1,param2:lr2"

            try:
                tupels = args.learning_rate.split(',')
                self.learning_rate = {param: float(lr)
                        for (param, lr) in map(lambda t: t.split(':'), tupels)}

            except ValueError:
                raise ValueError('Wrong format of learning-rate parameter.')

#         if not 'default' in self.learning_rate:
#             raise ValueError('learning rate does not contain "default"')

        self.learning_method = args.learning_method
        self.lr_adaptation_method = args.lr_adaptation_method
        self.epochs_limit = args.epochs_limit
        self.examples_limit = args.examples_limit
        self.store_params = args.store_params
        self.load_params = args.load_params
        self.out_dir = args.out_dir

        if args.hidden_layers:
            if not args.hidden_layers.strip():
                self.hidden_layers = []
            else:
                self.hidden_layers = [int(x)
                        for x in args.hidden_layers.strip().split(',')]

    def dump(self, model, outstream=None):
        filename = os.path.join(self.out_dir, self.get_name())
        state = pickle.dumps(model, pickle.HIGHEST_PROTOCOL)

        if outstream is not None:
            outstream.write(state)
            filename = outstream
        else:
            with open(filename + '.model', 'wb') as fh:
                fh.write(state)

        if self.store_params is not None:
            self.model.store_params(filename + u'.params', self.store_params,
                    True)

        return filename

    def dump_ready(self, epoch_finished):
        """ Return true if the network model that is being trained should be 
        dumped. It guarantees that we are dumping the network model periodically
        depending on self.dump_period
        """

        if epoch_finished and self.dump_each_epoch:
            return True

        if self.dump_period == -1:
            return False

        if self.period_type == 'time':
            return self.dump_ready_time()
        else: return self.dump_ready_examples()

        return False

    def dump_ready_time(self):
        """ Returns true if we should run the network over the training dataset.
        It guarantees that we are calculating our training loss periodically
        depending on self.validation_period/4
        """

        if self.period_has_passed(time(), self._last_dump, self.dump_period):
            self._last_dump = time()
            return True
        return False

    def dump_ready_examples(self):
        """ Returns true if we should run the network over the training dataset.
        It guarantees that we are calculating our validation loss periodically
        depending on self.validation_period/4
        """

        if self.period_has_passed(self.model.total_examples, self._last_dump,
                self.dump_period):
            self._last_dump = self.model.total_examples
            return True
        return False

    def dump_statistics(self):
        self.train_avg_cost_series.append(self.last_train_avg_cost())
        self.time_series.append(time() - self.start_time)
        self.train_total_examples.append(self.model.total_examples)

        stats = {'time_series': self.time_series,
                'examples_series': self.train_total_examples,
                'train_loss_series': self.train_avg_cost_series}

        if self.validation_file:
            self.dev_avg_cost_series.append(self.last_dev_avg_cost())
            stats['dev_loss_series'] = self.dev_avg_cost_series

        json.dump(stats, open('stats.json', 'wb'))

    def early_exit(self, early_stopping):
        """ Decides if the training should finish before completing all the
        epochs. This is handy if we want to avoid over-fitting.
        """

        if early_stopping:
            return True

        if self.epochs_limit == -1:
            epoch_criteria = False
        else:
            epoch_criteria = self.model.total_epochs >= self.epochs_limit

        if self.examples_limit == -1:
            examples_criteria = False
        else:
            examples_criteria = self.model.total_examples >= self.examples_limit

        if epoch_criteria or examples_criteria:
            return True
        return False

    def epoch_finished(self):
        """
        Handle things to be done after one epoch has finished, e.g., log.
        """
        super(MiniBatchTrainer, self).epoch_finished()
        self.model.total_epochs += 1
        log.info('finished epoch %d, handled %d instances in total' %
                (self.model.total_epochs, self.model.total_examples))

    def exit_train(self):
        log.info('Training has exited...')
        log.info('Finished %d epochs', self.model.total_epochs)
        log.info('Finished %d examples', self.model.total_examples)
        log.info('storing last model')
        self.dump(self.model)

        if not self.dump_best:
            return

        if self.best_score == None and self.validation_file:
            log.info('No best score found. (No validations completed?). ' +
                    'Forcing a validation step!')

            # FIXME: In the weird case that neither model was ever better than
            # 1e20 (default value in validate()), there is no best model to be
            # dumped here.
            self.validate()

        log.info('Saving best model found during training, score: %f' %
                self.best_score)

        filename = 'best_%s' % self.best_name
        filename = os.path.join(self.out_dir, filename)

        file_writer = open(filename, 'wb')
        self.best_model.seek(0)
        file_writer.write(self.best_model.read())
        file_writer.close()

    def get_model(self):
        # Don't need to build the model again, after loading it.
        if self.load_model:
            self.model = load_object_from_file(self.load_model)
            self.skip_examples = self.model.total_examples
            # Don't use the stored learning rate, but the one the user provided.
            # Problem with that: Sometimes we don't want that. Especially when
            # the model should proceed the training with learning rate it has
            # when it stopped training.
#             self.model.set_learning_rate(float(self.learning_rate),
#                     self.learning_method, self.lr_adaptation_method)

            if self.period_type == 'time':
                self._last_validation = time()
            else:
                self._last_validation = self.model.total_examples

        else:
            self.model = self.create_model()
            self.model.set_learning_rate(self.learning_rate,
                    self.learning_method, self.lr_adaptation_method)

        if self.load_params is not None:
            self.model.load_params(self.load_params[0], self.load_params[1])

        self.model.link(self.inputs)
        self.model.build()

    def last_train_avg_cost(self):
        processed_batches = self.train_total_batches - self.train_monitor['batches']
        accumulative_cost = self.model.total_costs - self.train_monitor['costs']
        avg_cost = accumulative_cost / processed_batches
        self.train_monitor['batches'] = self.train_total_batches
        self.train_monitor['costs'] = self.model.total_costs
        return avg_cost

    def last_dev_avg_cost(self):
        processed_batches = self.dev_total_batches - self.dev_monitor['batches']
        accumulative_cost = self.dev_total_costs - self.dev_monitor['costs']
        avg_cost = accumulative_cost / processed_batches
        self.dev_monitor['batches'] = self.dev_total_batches
        self.dev_monitor['costs'] = self.dev_total_costs
        return avg_cost

    def period_has_passed(self, total, previous, period):
        """Indicate whether or not a report is due.

        Parameters
        ----------
        total : int
            total time or number of examples that have been handled
        previous : int
            time or number of examples handled since last report
        period : int
            time or number of examples that have to be handled in order to
            allow a new report

        Returns
        -------
        bool
            True if a new report is due, False otherwise
        """
        return total - previous >= period

    def remaining(self, progress=0.0):
        total_progress = self.model.total_epochs + progress
        epoch_remaining = (1.0 - (total_progress / self.epochs_limit))
        example_remaining = (1.0 - (float(self.model.total_examples) /
                self.examples_limit))

        if self.epochs_limit == -1:
            return example_remaining
        elif self.examples_limit == -1:
            return epoch_remaining

        return min(epoch_remaining, example_remaining)

    def _do_skip_examples(self):
        """Skip some examples after model loading.

        We need that when we load a stored model to be able to continue the
        training where we left.
        Caution: This is of course a naive implementation, because it may run
        over the entire dataset multiple times. But don't try to store the
        number of examples per epoch and then just start at the beginning of the
        dataset, because the batches might cross epochs' boundaries.

        Returns
        -------
        generator
            Batch generator that proceeds at the position where the loaded model
            stopped. If no examples must be skipped, it returns a new generator.
        """
        batch_generator = self.next_batch(self.train_file)

        while self.skip_examples > 0:
            (batch, epoch_finished) = batch_generator.next()

            # Reopen the training file
            if epoch_finished:
                batch_generator = self.next_batch(self.train_file)

            self.skip_examples -= len(batch)

        return batch_generator

    def run(self):
        self.before_run_begins()

#         printing.pydotprint(self.model.trainer, outfile='trainer.png')
#         theano.printing.pydotprint(self.model.validator, outfile='validator.png')

        self.start_time = time()
        example_count_since_validation, costs_since_validation = 0, 0.0

        theano_processing = 0.0
        t = time()

        batch_generator = self._do_skip_examples()

        while True:
            log_iterations(log, self.train_total_batches, 10000)

            (batch, epoch_finished) = batch_generator.next()
            arguments = self.prepare_arguments(batch)

            t_start = time()
            output = self.model.trainer(*arguments)
            theano_processing += time() - t_start

            cost = output[0]
            self.model.total_examples += len(batch)
            self.model.total_costs += float(cost)
            self.train_total_batches += 1
            costs_since_validation += float(cost)
            example_count_since_validation += len(batch)

            self.model.update_learning_rate(self.remaining())

            if epoch_finished:
                batch_generator = self.next_batch(self.train_file)
                self.epoch_finished()

            if self.dump_ready(epoch_finished):
                self.dump(self.model)

            early_stopping = False

            if self.validation_ready():
                # report training error
                t = float(time() - t)
                avg_cost = costs_since_validation / \
                        float(example_count_since_validation)
                log.info('Average loss on %d example of the training set is %f',
                         example_count_since_validation, avg_cost)
                log.info('Speed of training is %f example/s',
                         example_count_since_validation / t)
                log.info('Percentage of time spent by theano processing is %f',
                         theano_processing / t)
                log.info('Processed %d so far.', self.model.total_examples)

                example_count_since_validation, costs_since_validation = 0, 0.0

                early_stopping = self.validate()

                t = time()
                theano_processing = 0.0

            if self.early_exit(early_stopping):
                break

        self.exit_train()

    def set_period_type(self, type_):
        """ Set frequency of reports depending on number of examples or time."""
        if type_ == 'time':
            self._last_validation = time()
            self._last_dump = time()
        elif type_ == 'examples':
            self._last_validation = 0
            self._last_dump = 0

    def validate(self):
        """ Calculates the average loss over the validation set.

        Returns
        -------
        bool
            False: early stopping criterion did not fire
            True: early stopping criterion fired (you should stop the training)
        """

        if not self.validation_file:
            return False

        log.info('start validation on validation sets')
        main_avg_cost = np.iinfo(np.int).max
        previous_validation = (self.previous_validation_avg_cost,
                self.previous_validation_ppl)

        for i, f in enumerate(self.validation_file):
            avg_cost, ppl = self._validate_single_file(f)

            if i != 0:
                continue

            # Only do this with the first validation file
            main_avg_cost = avg_cost
            self.previous_validation_avg_cost = main_avg_cost
            self.previous_validation_ppl = ppl

            model_is_worse = \
                    (self.perplexity and previous_validation[1] < ppl) or \
                    (not self.perplexity and
                    previous_validation[0] < main_avg_cost)

            if model_is_worse and self.lr_adaptation_method == 'MniTeh12':
                # If we have only one learning rate for all parameters
                # reduce only this. Otherwise, decrease all existing
                # learning rates.
                new_lr = {k: v * 0.5 for k, v in
                        self.model.get_learning_rate().iteritems()}

                log.debug('validation error/perplexity rose, adapting ' + \
                        'learning rate to %s' % str(new_lr))
                self.model.set_learning_rate(new_lr)

        # Is early stopping active?
        if self.early_stopping != -1:

            if model_is_worse:
                self._early_stopping_count += 1
            else:
                self._early_stopping_count = 0

            if self._early_stopping_count >= self.early_stopping:
                return True

        if self.dump_best:
            error_measure = self.previous_validation_ppl \
                    if self.perplexity else main_avg_cost
            self.dump_best_model(error_measure)

        return False

    def _validate_single_file(self, filename):
        """Validate the model in the given file.

        Parameters
        ----------
        filename : str
            name of the validation file

        Returns
        -------
        float
            avg cost per example
        float
            perplexity or np.nan depending on whether self.perplexity is set
        """
        t = time()
        costs = 0.
        log_probabs = 0.
        num_examples = 0
        num_ppl_examples = 0
        log.info('validation on file: %s' % filename)
        batch_generator = self.next_batch(filename)

        while True:
            batch, epoch_finished = batch_generator.next()
            arguments = self.prepare_arguments(batch)
            cur_cost, similarities = \
                    self.model.validator(*arguments)[:2]
            costs += cur_cost

            # Compute cost and perplexity using a different function.
            if self.perplexity:
                _, _, cur_log_percents, cur_num_ppl_examples = \
                        self._calc_probabilities_from_similarity(arguments[1],
                        similarities)
                log_probabs += cur_log_percents if cur_log_percents is not np.nan else 0.
                num_ppl_examples += cur_num_ppl_examples if cur_log_percents is not np.nan else 0

            num_examples += len(batch)

            if epoch_finished:
                break

        t = time() - t
        self.dev_total_costs += costs
        avg_cost = costs / num_examples
        log.info('Average loss on %d example of the validation set is %f',
                num_examples, avg_cost)

        ppl = np.nan

        if self.perplexity:
            ppl = np.exp(-1. / num_ppl_examples * log_probabs)
            log.info('Perplexity on %d example of the validation set is %f',
                    num_ppl_examples, ppl)

        log.info('Speed of validation is %f example/s', num_examples / t)
        return avg_cost, ppl

    def validation_ready(self):
        """ Return true if the network model that is being trained should be
        dumped. It guarantees that we are dumping the network model periodically
        depending on self.dump_period
        """
        if self.validation_period == -1:
            return False

        if self.period_type == 'time':
            return self.validation_ready_time()
        else: return self.validation_ready_examples()

        return False

    def validation_ready_examples(self):
        """ Returns true if we should run the network over the validation 
        dataset. It guarantees that we are calculating our validation loss 
        periodically depending on self.validation_period
        """

        if self.period_has_passed(self.model.total_examples,
                self._last_validation, self.validation_period):
            self._last_validation = self.model.total_examples
            return True

        return False

    def validation_ready_time(self):
        """ Returns true if we should run the network over the validation 
        dataset. It guarantees that we are calculating our validation loss
        periodically depending on self.validation_period
        """

        if self.period_has_passed(time(), self._last_validation,
                self.validation_period):
            self._last_validation = time()
            return True

        return False


class EmbeddingsMiniBatchTrainer(MiniBatchTrainer):
    """This class is being subclassed by Models using embeddings.

    It provides functionality of automatic embeddings and vocabulary storing.
    """

    def __init__(self):
        super(EmbeddingsMiniBatchTrainer, self).__init__()
        self.do_dump_vocabulary = True
        self.do_dump_embeddings = True

    def before_run_begins(self):
        super(EmbeddingsMiniBatchTrainer, self).before_run_begins()

        if self.do_dump_vocabulary:
            self.dump_vocabulary()

        if self.do_dump_embeddings:
            self.dump_embeddings()

    def configure(self, args):
        super(EmbeddingsMiniBatchTrainer, self).configure(args)
        self.vocab = read_vocabulary_id_file(args.vocabulary)
        self.vocab_size = len(self.vocab.keys())
        self.effective_vocab_size = len(self.vocab.keys())
        self.word_embedding_size = args.word_embedding_size
        self.do_dump_vocabulary = args.dump_vocabulary
        self.do_dump_embeddings = args.dump_embeddings
        log.debug('Effective size of the vocabulary %d',
                self.effective_vocab_size)

    def dump(self, model, outstream=None):
        name = self.get_name()

        if self.do_dump_embeddings:
            self.dump_embeddings('%s.embeddings' % name)

        return super(EmbeddingsMiniBatchTrainer, self).dump(model, outstream)

    def dump_embeddings(self, filename=None, embeddings=None):
        """Dump the word embeddings."""

        if filename is None:
            filename = '%s_%d.embeddings' % (self.name, self.model.total_epochs)

        if embeddings is None:
            embeddings = self.model.get_word_embeddings()

        np.savetxt(filename, embeddings)

    def dump_vocabulary(self, filename=None, vocabulary=None):
        """Dump the vocabulary after importing it.

        This might be useful, if the originally provided vocabulary does contain
        duplicates.
        """

        if filename is None:
            filename = os.path.join(self.out_dir, '%s.vocab' % self.name)


        if vocabulary is None:
            vocabulary = self.vocab

        write_vocabulary_file(filename, vocabulary)

    def get_embedding_layer(self, effective_vocab, size_embedding, name='C'):
        embedding_layer = EmbeddingLayer(name,
                (effective_vocab, size_embedding))

        log.debug('Created new embedding layer named %s, size: (%d x %d)',
                name, effective_vocab, size_embedding)
        return embedding_layer


class HingeMiniBatchTrainer(EmbeddingsMiniBatchTrainer):

    def __init__(self):
        super(HingeMiniBatchTrainer, self).__init__()

        self.observed = T.matrix('observed', dtype=intX)
        self.corrupted = T.matrix('corrupted', dtype=intX)
        self.inputs = [self.observed, self.corrupted]

    def configure(self, args):
        super(HingeMiniBatchTrainer, self).configure(args)
        self.left_context = args.left_context
        self.right_context = args.right_context

    def create_model(self):
        """Create the new network that is supposed to be trained.

        Returns
        -------
        network
            network to be trained by this trainer
        """
        emb_matrix_shape = (self.effective_vocab_size, self.word_embedding_size)
        no_of_tokens = self.left_context + self.right_context + 1
        return WordPhraseNetwork(emb_matrix_shape=emb_matrix_shape,
                no_of_tokens=no_of_tokens, hidden_layers=self.hidden_layers)

    def process_example(self, example):
        """Create a fake example out of the given correct one.

        Additionally, replaces invalid token ids by the unknown word.
        """
        example = text_to_vocab_indices(self.effective_vocab_size, example)[0]

        fake_example = [x for x in example]
        fake_example[self.left_context] = randint(0, self.effective_vocab_size - 1)
        return (example, fake_example)


class HingeSentimentMiniBatchTrainer(HingeMiniBatchTrainer):
    """Create a trainer that produces sent_1 embeddings.

    The negative example is created by replacing the sentiment word by a
    random word from the entire vocabulary.
    """
    example_iterator_type = SentimentExamplesGenerator

    def configure(self, args):
        self.sent_vocab = set(read_vocabulary_id_file(args.sent_vocab, False))
        super(HingeSentimentMiniBatchTrainer, self).configure(args)


class HingeSentiment2MiniBatchTrainer(HingeSentimentMiniBatchTrainer):
    """Create a trainer that produces sent_2 embeddings.

    The negative example is created by replacing the sentiment word by another
    random sentiment word from the entire vocabulary.
    """

    def choose_random_sentiment_word(self):
        """Return a random sentiment word's index in the global vocabulary.

        I.e., choose a random sentiment word from the sentiment vocabulary and
        return this word's index in the globel vocabulary.

        Returns
        -------
        int
            index of the sentiment word in the global vocabulary
        """
        sent_word = choice(self.sent_vocab.keys())
        raise ValueError(
                'Here is a fixme that needs to be checked before proceeding.')

        # FIXME: should this be: self.vocab.get(sent_word, Token.UNKNOWNS)???
        return self.vocab.get(sent_word, SpecialTokenID.UNKNOWN.value)

    def process_example(self, example):
        example = text_to_vocab_indices(self.effective_vocab_size, example)[0]
        fake_example = [x for x in example]
        fake_example[self.left_context] = self.choose_random_sentiment_word()
        return (example, fake_example)


class MlpTrainer(MiniBatchTrainer):
    """Create a multi-layer perceptron."""
    example_iterator_type = LabeledExampleGenerator

    def __init__(self):
        super(MlpTrainer, self).__init__()
        self.input_indices = T.matrix('input_indices', dtype=floatX)
        self.label = T.matrix('label', dtype=floatX)
        self.inputs = [self.label, self.input_indices]

    def configure(self, args):
        super(MlpTrainer, self).configure(args)
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.error_func = args.error_func

    def create_model(self):
        return MultiLayerPerceptron(input_size=self.input_size,
                output_size=self.output_size, hidden_layers=self.hidden_layers,
                error_func=self.error_func)

    def process_example(self, example):
        """Convert the given example in handable data structures.

        Splits vectors in their single values and converts the labels into ints
        and the data into floats.

        Returns
        -------
        list(float)
            output (label)
        list(float)
            input
        """
        (label, data) = example
        return (map(float, label.split(' ')), map(float, data.split(' ')))


class SimpleVLblNceTrainer(EmbeddingsMiniBatchTrainer):
    """Create a vLBL NCE model."""

    def __init__(self):
        super(SimpleVLblNceTrainer, self).__init__()
        self.h_indices = debug_print(T.imatrix('h'), 'h_indices')
        self.w_indices = debug_print(T.ivector(name='w'), 'w_indices')
        self.inputs = [self.h_indices, self.w_indices]

    def configure(self, args):
        super(SimpleVLblNceTrainer, self).configure(args)

        if args.pred_vocab:
            # Element i contains the index of the i'th prediction vocabulary
            # token in the original vocabulary.
            self.vocab_mapping_list = list()

            # Mapping from the model vocabulary to the prediction vocabulary
            # indices
            self.vocab_mapping = dict()
            pred_vocab = read_vocabulary_file(args.pred_vocab, False)

            for i, token in enumerate(pred_vocab):

                if not token in self.vocab:
                    raise ValueError('Token "%s" in prediction vocabulary ' +
                            'does not exist in model vocabulary.' % token)

                self.vocab_mapping_list.append(self.vocab[token])
                self.vocab_mapping[self.vocab[token]] = i
        else:
            self.vocab_mapping_list = range(len(self.vocab))
            self.vocab_mapping = dict(
                    zip(self.vocab_mapping_list, self.vocab_mapping_list))

        self.left_context = args.left_context
        self.right_context = args.right_context
        self.k = args.noise_samples
        self.unigram = read_unigram_distribution(args.unigram)

        if self.unigram.shape[0] != self.effective_vocab_size:
            raise ValueError('unigram size is unequal to vocab size (%d / ' +
                    '%d); Have you added counts for special tokens?' %
                    (self.unigram.shape[0], self.effective_vocab_size))

        self.l1_weight = args.l1_weight
        self.l2_weight = args.l2_weight
        self.nce_seed = args.nce_seed

    def create_model(self):
        """Create the new network that is supposed to be trained.

        Returns
        -------
        network
            network to be trained by this trainer
        """
        return SimpleVLblNce(self.batch_size, self.effective_vocab_size,
                self.left_context, self.right_context,
                self.word_embedding_size, self.k, self.unigram,
                l1_weight=self.l1_weight, l2_weight=self.l2_weight,
                nce_seed=self.nce_seed)

    def dump_embeddings(self, filename=None, embeddings=None):
        """Dump the word embeddings."""

        if embeddings is not None:
            super(SimpleVLblNceTrainer, self).dump_embeddings(filename, embeddings)
            return

        if filename is None:
            filename = '%s_%d.embeddings' % (self.name, self.model.total_epochs)

        np.savetxt(filename + '_r', self.model.R.get_value())
        np.savetxt(filename + '_q', self.model.Q.get_value())

    def process_example(self, example):
        """Create a training example using the given tokens."""
        log.debug(example)
        example = text_to_vocab_indices(self.vocab, example)[0]
        log.debug(example)
        return (example[:self.left_context] if self.right_context == 0 else
                example[:self.left_context] + example[self.left_context + 1:],
                example[self.left_context])


class SimpleVLblNceSentimentTrainer(SimpleVLblNceTrainer):
    """Create a vLBL model that trains special sentiment embeddings."""
    example_iterator_type = SentimentAnywhereExamplesGenerator

    def __init__(self):
        super(SimpleVLblNceSentimentTrainer, self).__init__()
        self.handled_context_ids = set()
        self.handled_target_ids = set()

    def configure(self, args):
        self.sent_vocab = set(read_vocabulary_file(args.sent_vocab, False))
        super(SimpleVLblNceSentimentTrainer, self).configure(args)

    def dump(self, model, outstream=None):
        ret = super(SimpleVLblNceSentimentTrainer, self).dump(model, outstream)

        if self.do_dump_vocabulary:
            self.dump_handled_ids()

        return ret

    def dump_handled_ids(self):
        """
        Dumps the content of the two sets, which contain the ids of all seen
        tokens.
        """
        name = self.get_name()
        filename = '%s.handled_context' % name
        filename = os.path.join(self.out_dir, filename)
        self._dump_handled_ids(self.handled_context_ids, filename)

        filename = '%s.handled_target' % name
        filename = os.path.join(self.out_dir, filename)
        self._dump_handled_ids(self.handled_target_ids, filename)

    def _dump_handled_ids(self, ids, filename):

        with utf8_file_open(filename, 'w') as id_file:
            id_file.write('\n'.join([unicode(i) for i in ids]))

    def process_example(self, example):
        """Create a training example using the given tokens.

        Keep track of vocabulary items we have seen during the training.
        """
        example = super(SimpleVLblNceSentimentTrainer, self).process_example(example)
        self.handled_context_ids |= set(example[0])
        self.handled_target_ids.add(example[1])
        return example


class VLblNceTrainer(SimpleVLblNceTrainer):

    def create_model(self):
        """Create the new network that is supposed to be trained.

        Returns
        -------
        network
            network to be trained by this trainer
        """
        return VLblNce(self.batch_size, self.effective_vocab_size,
                self.left_context, self.right_context,
                self.word_embedding_size, self.k, self.unigram,
                l1_weight=self.l1_weight, l2_weight=self.l2_weight,
                nce_seed=self.nce_seed)


class VLblNceSentimentTrainer(SimpleVLblNceSentimentTrainer):
    """Create a vLBL model that trains special sentiment embeddings."""

    def create_model(self):
        """Create the new network that is supposed to be trained.

        Returns
        -------
        network
            network to be trained by this trainer
        """
        return VLblNce(self.batch_size, self.effective_vocab_size,
                self.left_context, self.right_context,
                self.word_embedding_size, self.k, self.unigram,
                l1_weight=self.l1_weight, l2_weight=self.l2_weight,
                nce_seed=self.nce_seed)


class NvLblNceTrainer(SimpleVLblNceTrainer):

    def configure(self, args):
        self.activation_func = args.activation_func
        super(NvLblNceTrainer, self).configure(args)

    def create_model(self):
        """Create the new network that is supposed to be trained.

        Returns
        -------
        network
            network to be trained by this trainer
        """
        return NvLblNce(activation_func=self.activation_func,
                batch_size=self.batch_size,
                vocab_size=self.effective_vocab_size,
                left_context=self.left_context,
                right_context=self.right_context,
                emb_size=self.word_embedding_size,
                k=self.k,
                unigram=self.unigram,
                l1_weight=self.l1_weight, l2_weight=self.l2_weight,
                nce_seed=self.nce_seed)


class LblNceTrainer(SimpleVLblNceTrainer):

    def create_model(self):
        """Create the new network that is supposed to be trained.

        Returns
        -------
        network
            network to be trained by this trainer
        """
        return LblNce(batch_size=self.batch_size,
                vocab_size=self.effective_vocab_size,
                left_context=self.left_context,
                right_context=self.right_context,
                emb_size=self.word_embedding_size,
                k=self.k,
                unigram=self.unigram,
                l1_weight=self.l1_weight, l2_weight=self.l2_weight,
                nce_seed=self.nce_seed)


class NlblNceTrainer(SimpleVLblNceTrainer):

    def configure(self, args):
        self.activation_func = args.activation_func
        super(NlblNceTrainer, self).configure(args)

    def create_model(self):
        """Create the new network that is supposed to be trained.

        Returns
        -------
        network
            network to be trained by this trainer
        """
        return NlblNce(activation_func=self.activation_func,
                batch_size=self.batch_size,
                vocab_size=self.effective_vocab_size,
                left_context=self.left_context,
                right_context=self.right_context,
                emb_size=self.word_embedding_size,
                k=self.k,
                unigram=self.unigram,
                l1_weight=self.l1_weight, l2_weight=self.l2_weight,
                nce_seed=self.nce_seed)


class SLmNceTrainer(SimpleVLblNceTrainer):

    def configure(self, args):
        self.activation_func = args.activation_func
        super(SLmNceTrainer, self).configure(args)

    def create_model(self):
        """Create the new network that is supposed to be trained.

        Returns
        -------
        network
            network to be trained by this trainer
        """
        return SLmNce(hidden_neurons=self.hidden_layers[0],
                activation_func=self.activation_func,
                batch_size=self.batch_size,
                vocab_size=self.effective_vocab_size,
                left_context=self.left_context,
                right_context=self.right_context,
                emb_size=self.word_embedding_size,
                k=self.k,
                unigram=self.unigram,
                l1_weight=self.l1_weight, l2_weight=self.l2_weight,
                nce_seed=self.nce_seed)


class VLblNceDistributionalTrainer(VLblNceTrainer):

    def create_model(self):
        """Create the new network that is supposed to be trained.

        Returns
        -------
        network
            network to be trained by this trainer
        """
        return VLblNceDistributional(self.batch_size, self.effective_vocab_size,
                self.left_context, self.right_context,
                self.word_embedding_size, self.k, self.unigram,
                l1_weight=self.l1_weight, l2_weight=self.l2_weight,
                nce_seed=self.nce_seed)
