# -*- coding: utf-8 -*-
"""
This file contains text related utility functions.
"""
from nltk.tokenize import word_tokenize

def tokenize(text):
    """Tokenize the given input file by NLTK\'s recommended word tokenizer.

    Parameters
    ----------
    text : str
        text to tokenize

    Returns
    -------
    list(str)
        tokenized text
    """
    return word_tokenize(text.strip())
