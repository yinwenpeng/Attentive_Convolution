# -*- coding: utf-8 -*-
"""
This file contains classes dealing with labeled data.
"""
from logging import getLogger

from scipy.stats._rank import rankdata
from theano import tensor as T

from cis.deep.utils import load_object_from_file, text_to_vocab_indices, \
    file_line_generator, sort_dict_by_label, utf8_file_open, log_iterations, \
    ndarray_to_string
from cis.deep.utils import logger_config
from cis.deep.utils.embeddings import read_vocabulary_id_file
from cis.deep.utils.theano import debug_print
import numpy as np
from word2embeddings.nn.layers import floatX, intX
from word2embeddings.nn.trainer import MiniBatchRunner
from word2embeddings.tools.examples_generator import PlainExampleGenerator, \
    PaddedWindowExamplesGenerator


log = getLogger(__name__)
logger_config(log)

class MiniBatchPredictor(MiniBatchRunner):
    """Base class for predictors that work in mini batches."""
    example_iterator_type = PlainExampleGenerator

    def __init__(self):
        super(MiniBatchPredictor, self).__init__()

    def configure(self, args):
        super(MiniBatchPredictor, self).configure(args)
        self.predict_file = args.predict_file

    def get_model(self):
        self.model = load_object_from_file(self.load_model)
        self.predictor_method = self.model.predictor

    def run(self):
        """Predict the output of the model on all test examples.

        Returns
        -------
        list
            list of predicitons
        """
        predictions = []

        for example in self.epoch_iter(self.predict_file):
            predictions.append(self.model.predictor([example]))

        return predictions

    def predict_single(self):
        """Predict the output of the model on all test examples yielding one
        example at a time.

        Returns
        -------
        list
            list of results for the current example
        """
        for example in self.epoch_iter(self.predict_file):
            yield self.predictor_method([example])


class LblPredictor(MiniBatchPredictor):

    def __init__(self):
        super(LblPredictor, self).__init__()
        self.input_data = T.matrix('input_data', dtype=floatX)
#         self.label = T.matrix('label', dtype=intX)
        self.inputs = [self.input_data]

    def configure(self, args):
        super(LblPredictor, self).configure(args)
        self.vocab = read_vocabulary_id_file(args.vocabulary)
        self.vocab_size = len(self.vocab.keys())
        self.effective_vocab_size = len(self.vocab.keys())

    def process_example(self, example):
        """Convert the given example in handable data structures.

        Splits vectors in their single values and converts the labels into ints
        and the data into floats.

        Returns
        -------
        list(str)
            input text
        """
#         return example.split(' ')
        return text_to_vocab_indices(self.vocab, example)[0]


class vLblNCEPredictor(MiniBatchPredictor):
    def __init__(self):
        super(vLblNCEPredictor, self).__init__()
        self.h_indices = debug_print(T.imatrix('h'), 'h')
        self.inputs = [self.h_indices]

    def configure(self, args):
        super(vLblNCEPredictor, self).configure(args)
        self.vocab = read_vocabulary_id_file(args.vocabulary)
        self.vocab_size = len(self.vocab.keys())
        self.effective_vocab_size = len(self.vocab.keys())
        self.perplexity = args.perplexity
        self.save_word = args.save_word
        self.result_file = args.result_file
        self.store_rank = args.store_rank
        self.store_argmax = args.store_argmax
        self.store_softmax = args.store_softmax
        self.normalize_with_root = args.normalize_with_root
        self.information = args.information
        self.predictions = args.predictions

        # This code is taken from SimpleVLblNceTrainer
        if args.pred_vocab:
            # Element i contains the index of the i'th prediction vocabulary
            # token in the original vocabulary.
            self.vocab_mapping_list = list()

            # Mapping from the model vocabulary to the prediction vocabulary
            # indices
            self.vocab_mapping = dict()

            for i, token in enumerate(file_line_generator(args.pred_vocab)):

                if not token in self.vocab:
                    raise ValueError('Token "%s" in prediction vocabulary ' +
                            'does not exist in model vocabulary.' % token)

                self.vocab_mapping_list.append(self.vocab[token])
                self.vocab_mapping[self.vocab[token]] = i
        else:
            self.vocab_mapping_list = range(len(self.vocab))
            self.vocab_mapping = dict(
                    zip(self.vocab_mapping_list, self.vocab_mapping_list))

        if self.perplexity:
            self.example_iterator_type = PaddedWindowExamplesGenerator
            self.example_processor = self._process_example_full_text
            self.learn_eos = True  # We need to set that because otherwise PaddedWindowExampleGenerator will ignore end-of-sentence tags (</S>)
            self.disable_padding = False
            self.w_indices = debug_print(T.imatrix('w'), 'w')
            self.inputs.append(self.w_indices)
        else:
            self.example_processor = self._process_example_context_per_line

    def get_model(self):
        super(vLblNCEPredictor, self).get_model()

        if self.perplexity:
            self.left_context = self.model.left_context
            self.right_context = self.model.right_context

    def predict_single(self):
        """Predict the output of the model on all test examples yielding one
        example at a time.

        Returns
        -------
        list
            list of results for the current example
        """
        for example in self.epoch_iter(self.predict_file):
            example = [example]

            if self.perplexity:
                # Pass only the context, not the target word
                yield example, self.predictor_method(zip(*example)[0])
            else:
                yield example, self.predictor_method(example)

    def process_example(self, example):
        """Convert the given example in handable data structures.

        Splits vectors in their single values and converts the labels into ints
        and the data into floats.

        Returns
        -------
        list(str)
            input text
        """
        log.debug(example)
        res = self.example_processor(example)
        log.debug(res)
        return res[0]

    def _process_example_context_per_line(self, example):
        """Process the given example that contains only the context and not the
        target word.
        """
        return text_to_vocab_indices(self.vocab, example)

    def _process_example_full_text(self, example):
        """Process the given example that contains context and target word.

        The implementation is taken from SimpleVLblNceTrainer.process_example.
        """
        idx, example = text_to_vocab_indices(self.vocab, example)
        return (idx[:self.model.left_context] if self.model.right_context == 0 else
                idx[:self.model.left_context] + idx[self.model.left_context + 1:],
                idx[self.model.left_context]), example

    def run(self):
        vocab = dict(self.vocab)

        # Get a mapping from index to word
        vocab_entries = sort_dict_by_label(vocab)
        vocab_entries = zip(*vocab_entries)[0]
        log_probabs = 0.
        num_ppl_examples = 0
        num_examples = 0

        with utf8_file_open(self.result_file, 'w') as outfile:

            for batch, _ in self.next_batch(self.predict_file):
            # Handle each prediction
#             for (cur_count, (example, predictions)) in enumerate(self.predict_single()):

                log_iterations(log, num_examples, 10000)
                num_examples += len(batch)

                if self.perplexity:
                    batch = zip(*batch)
                    # Pass only the context, not the target word
                    predictions = self.predictor_method(batch[0])
                else:
                    self.predictor_method(batch)

                if self.store_softmax or self.store_rank or self.store_argmax \
                        or self.information or self.perplexity:
                    sm, probabs, cur_log_probabs, cur_num_ppl_examples = \
                            self._calc_probabilities_from_similarity(batch[1], predictions[1])
                    num_ppl_examples += cur_num_ppl_examples

                if self.store_rank or self.information:
                    # rankdata sorts ascending, i.e., distances, but we have
                    # similarities, hence, 1-sm
                    ranks = rankdata(1 - sm, method='min').astype(int)

                    if self.store_rank:
                        outfile.write(ndarray_to_string(ranks))

                    if self.information:
                        unique_ranks = set(ranks)
                        hard_idx = vocab[u'hard']
                        sorted_unique_ranks = ' '.join(map(str, sorted(unique_ranks)))
                        sorted_unique_ranks = ''
                        top_ten_entries = ' '.join([vocab_entries[i] for i in np.argsort(1 - sm)[:10]])
                        print '#%d\t%s\t%s' % (ranks[hard_idx],
                                sorted_unique_ranks,
                                top_ten_entries)

                if self.store_argmax:
                    maximum = np.argmax(sm)
    #                 outfile.write(vocab_entries[maximum] + u' (%d)\t' % maximum)
                    outfile.write(vocab_entries[maximum])

                if self.store_softmax:

                    if self.normalize_with_root:
                        sm = np.sqrt(sm)
                        sm = sm / np.linalg.norm(sm, 2, axis=-1)

                    outfile.write(ndarray_to_string(sm))

                if self.perplexity:

                    if self.save_word:
                        indices_in_predict_vocab = [self.vocab_mapping[batch[1][i]] for i in range(len(batch[1]))]
                        indices_in_original_vocab = [self.vocab_mapping_list[i] for i in indices_in_predict_vocab]
                        words = [self.vocab.keys()[self.vocab.values().index(i)] for i in indices_in_original_vocab]

                        outfile.write( u'\n'.join("%s %s" % (x, y) for x, y in zip(map(unicode, probabs), words)) )
                    else:
                        outfile.write(u'\n'.join(map(unicode, probabs)))

                    log_probabs += cur_log_probabs if cur_log_probabs is not np.nan else 0.

                if self.predictions:
                    outfile.write(ndarray_to_string(predictions[0][0]))

                outfile.write(u'\n')

            # print all results
    #         for predictions in predictions:
    #             outfile.write(ndarray_to_string(predictions[0][0]) + u'\t')
    #
    #             if args.store_softmax:
    #                 outfile.write(ndarray_to_string(predictions[1][0]) + u'\t')
    #
    #             outfile.write(vocab_entries[predictions[2][0]] + u' (%d)' % predictions[2][0])
    #             outfile.write(u'\n')
    # #             outfile.write(unicode(predictions) + u'\n')
        if self.perplexity:
            ppl = np.exp(-1. / (num_ppl_examples) * log_probabs)
            log.info('Perplexity on %d examples is %f', num_ppl_examples, ppl)


class MlpPredictor(MiniBatchPredictor):

    def __init__(self):
        super(MlpPredictor, self).__init__()
        self.input_data = T.matrix('input_data', dtype=floatX)
        self.label = T.matrix('label', dtype=intX)
        self.inputs = [self.label, self.input_data]

    def process_example(self, example):
        """Convert the given example in handable data structures.

        Splits vectors in their single values and converts the labels into ints
        and the data into floats.

        Returns
        -------
        list(float)
            values
        """
        return map(float, example.split(' '))
