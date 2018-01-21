#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""examples_generator.py: Examples generator for language models training."""

from cis.deep.utils import file_line_generator
from cis.deep.utils.embeddings import SpecialToken


class Error(Exception):
    """ Base class to be used for other module's exceptions."""


class SpanNotUsedError(Error):
    """ Raised if the a token of a particular span is not picked."""


class ExampleGenerator(object):

    def configure(self, options):
        """Configure the example generator with the given options.

        Parameters
        ----------
        options : dict
            options dictionary
        """
        pass

    def example_iter(self, filename):
        """Iterate over the examples in the given file.

        Must be implemented by each sub class.

        Parameters
        ----------
        filename : str
            name of the file containing the examples

        Yields
        ------
        examples
        """
        raise NotImplementedError


class PlainExampleGenerator(ExampleGenerator):
    """Reads a file containing only examples.

    Each line is considered one example.
    """
    def example_iter(self, filename):

        for example in file_line_generator(filename):
            yield example

        raise StopIteration


class LabeledExampleGenerator(PlainExampleGenerator):
    """Reads a file containing label and data for every example.

    Each line in the file is an example. The first column corresponds to the
    label, the second column is the input of the classifier. Columns are tab-
    separated. Vector-based inputs or outputs are space separated. However, this
    generator does not convert the values in any way. Instead, it just returns
    the values as strings.
    """

    def example_iter(self, filename):

        for line in super(LabeledExampleGenerator, self).example_iter(filename):
            yield line.split('\t')

        raise StopIteration


class PaddedWindowExamplesGenerator(PlainExampleGenerator):
    """ Generates sequence of fixed-width window of tokens."""

    def configure(self, options):
        self.left_context = options.left_context
        self.right_context = options.right_context
        self.disable_padding = options.disable_padding
        self.learn_eos = options.learn_eos

    def example_iter(self, filename):

        for line in super(PaddedWindowExamplesGenerator, self).example_iter(filename):
            line = line.split()

            if not self.disable_padding:
                line = self.pad_sent(line)
            elif self.learn_eos:  # add eos token if we need to learn it, but do not want to do padding
                line.append(SpecialToken.SENT_END.value)

            for example in self.sent_examples(line):
                yield example

    def is_valid_example(self, _):
        """Checks if the given example is a valid example to process.

        Every subclass can specify what is a valid example.
        """
        return True

    def pad_sent(self, tokens):
        sent = [SpecialToken.SENT_START.value]
        sent.extend(tokens)
        sent.append(SpecialToken.SENT_END.value)
        return sent

    def sent_examples(self, sent):
        """Turns a sentence into a number of examples.
             An example is like {'sources': [list of feature vectors]}
        """
        length = len(sent)

        # if the padding is disabled start pos from leftcontext+1
        start_offset = self.left_context if self.disable_padding else 1
        end_offset = self.right_context if self.disable_padding else 1

        # if we want to learn end-of-sentence during padding,
        # then move end_offset to let pos cover eos token
        if not self.disable_padding and self.learn_eos:
            end_offset -= 1

        for pos in range(start_offset, length - end_offset):
            left_context = sent[max(0, pos - self.left_context): pos]
            right_context = sent[pos + 1: pos + 1 + self.right_context]

            left_diff = self.left_context - len(left_context)

            if left_diff > 0:
                left_context = left_diff * [SpecialToken.PAD.value] + \
                        left_context

            right_diff = self.right_context - len(right_context)

            if right_diff > 0:
                right_context = right_context + right_diff * \
                        [SpecialToken.PAD.value]

            example = left_context + [sent[pos]] + right_context

            if not self.is_valid_example(example):
                continue

            yield example


class SentimentExamplesGenerator(PaddedWindowExamplesGenerator):
    """Extract special sentiment training examples.

    Extracts positive instances from the text, having the requirement that
    the center word of the example is contained in a sentiment vocabulary.

    Attributes
    ----------
    vocab : dict
        sentiment vocabulary
    """

    def configure(self, options):
        """
        Parameters
        ----------
        options.vocab : dict
            sentiment vocabulary

        """
        super(SentimentExamplesGenerator, self).configure(options)
        self.sent_vocab = options.sent_vocab

    def is_valid_example(self, example):
        return example[self.left_context] in self.sent_vocab


class SentimentAnywhereExamplesGenerator(SentimentExamplesGenerator):
    """Extract special sentiment training examples.

    Extracts examples from the text, having the requirement that
    at least one token of the example is contained in a sentiment vocabulary.
    """

    def is_valid_example(self, example):
        return any((e in self.sent_vocab for e in example))
