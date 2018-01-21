#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Language modeling networks used by the trainer."""

import theano

from word2embeddings.nn.layers import EmbeddingLayer, HingeLayer
from word2embeddings.nn.networks import Network, StackedBiasedHidden


class WordPhraseNetwork(Network):
    """Model to distinguish between corrupted phrases and observed ones."""

    def __init__(self, name='WordPhrase', emb_matrix_shape=None, no_of_tokens=1,
                             hidden_layers=[1]):
        super(WordPhraseNetwork, self).__init__(name=name)
        _, word_size = emb_matrix_shape
        layers = [no_of_tokens * word_size]
        layers.extend(hidden_layers)
        layers.append(1)
        self.word_embedding = EmbeddingLayer(name='w_embedding',
                shape=emb_matrix_shape)
        self.hidden_stack = StackedBiasedHidden(name='w_stack', layers=layers)
        self.loss = HingeLayer(name='loss')

        self.layers = [self.word_embedding, self.hidden_stack, self.loss]

    def link(self, inputs):
        self.inputs = inputs
        observed_phrases = inputs[0]
        corrupted_phrases = inputs[1]
        observed_words = self.word_embedding.link([observed_phrases])[0]
        observed_scores = self.hidden_stack.link([observed_words])[0]
        corrupted_scores = theano.clone(observed_scores,
                {observed_phrases: corrupted_phrases})
        self.outputs = self.loss.link([observed_scores, corrupted_scores])
        return self.outputs

    def get_word_embeddings(self):
        return self.word_embedding.weights.get_value()
