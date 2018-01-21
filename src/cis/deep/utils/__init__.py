# -*- coding: utf-8 -*-
"""
This file contains common utility classes and methods.
"""
from bz2 import BZ2File
import cPickle
import codecs
import collections
from datetime import date
import hashlib
import io
from itertools import izip_longest
import logging
from operator import itemgetter
import os
import sys

import numpy as np
import itertools


def are_generators_equal(gen1, gen2):
    """Indicate whether or not the given generators are equal.

    Generators cannot be compared as easily as lists. Here's the description of
    what happens:
    "This can actually short-circuit without necessarily having to look at all
    values. As pointed out by larsmans in the comments, we can't use izip() here
    since it might give wrong results if the generators produce a different
    number of elements – izip() will stop on the shortest iterator. We use a
    newly created object instance as fill value for izip_longest(), since object
    instances are also compared by object identity, so sentinel is guaranteed to
    compare unequal to everything else."
    [http://stackoverflow.com/questions/9983547/comparing-two-generators-in-python]
    """
    return all(a == b for a, b in
                izip_longest(gen1, gen2, fillvalue=object()))

def digest(string):
    """Calculate a hash for the given string.

    Parameters
    ----------
    string : str
        string to calculate the hash for

    Examples
    --------
    >>> digest('hello world')
    '2f05477fc24bb4faefd86517156dafdecec45b8ad3cf2522a563582b'
    """
    return hashlib.sha224(string).hexdigest()

def file_line_generator(filename, strip=True, comment=None):
    """Iterates over the lines in a file.

    Each line is one string. Uses utf8_file_open.

    Parameters
    ----------
    filename : str
        name of the file to load
    strip : bool
        indicates whether or not to strip each line after reading (removes line
        endings, but also tabs or spaces at the beginning of the line)
    comment : str
        if a line in the file starts with this string, then it's considered to
        be a comment and discarded. None if nothing should be discarded.

    Returns
    -------
    list
        each line of the given file is one item in the list
    """

    with utf8_file_open(filename) as f:

        for line in f:

            if strip:
                line = line.strip()

            if comment and line.startswith(comment):
                continue

            yield line

    raise StopIteration

def flatten_iterable(it):
    """Flattens an iteratable object.

    Parameters
    ----------
    it : iterable
        nested iterable

    Returns
    -------
    generator
        generator that iterates over all items in the iterable
    """

    for item in it:
        if isinstance(item, collections.Iterable) and \
                not isinstance(item, basestring):

            for sub in flatten_iterable(item):
                yield sub
        else:
            yield item

def generator_has_next(gen):
    """Check if the given generator contains more elements.

    This is a hack. If the generator contains more elements, the returned
    generator must be used, because the original generator "lost" an element.
    The returned generator however contains this element. This is possible by
    using itertools.chain.

    Returns
    -------
    Any
        False: generator does not contain any more elements
        generator: generator does contain more elements, use this generator
        instead of the original one, otherwise you loose one element.
    """

    try:
        elem = gen.next()
        return itertools.chain([elem], gen)
    except StopIteration:
        return False


def load_object_from_file(filename):
    """Loads an object from the given filename.

    The given file must have been written using save_object.

    Parameters
    ----------
    filename : string
        name of the persisted object
    """
    # Caution: using utf8_file_open doesn't work with cPickle
    return cPickle.load(open(filename, 'rb'))

def log_iterations(log, count, log_every):
    """Log how many iterations have been handled every log_every iterations.

    Parameters
    ----------
    log : logger
        logger to be logged into
    count : int
        current count of iterations
    log_every : int
        the count is logged every log_every iterations
    """

    if count % log_every == 0:
        log.info('iterations: ' + str(count))


def logger_config(logger, level=logging.INFO, log_dir=None):
    """Configure the given logger.

    Parameters
    ----------
    logger : logger
        logger to configure
    log_dir : str
        path where to store the log file, if None no log file is created
    """
    logger.setLevel(level)
    formatter = _logger_config_create_formatter()
    logger.addHandler(_logger_config_create_console_handler(formatter, level))

    if log_dir is not None:
        logger.addHandler(_logger_config_create_file_handler(formatter, level,
                log_dir))

def _logger_config_create_formatter():
    """Return a formatter object."""
    formatter = logging.Formatter(
            '%(asctime)s\t%(levelname)s\t%(module)s\t%(funcName)s\t%(message)s',
            '%Y-%m-%d %H:%M:%S')
    return formatter

def _logger_config_create_console_handler(formatter, level):
    """Return a console handler."""
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    return ch

def _logger_config_create_file_handler(formatter, level, log_dir):
    """Return a log file handler."""
    fh = logging.FileHandler(os.path.join(log_dir, 'log-' +
            date.today().strftime('%Y-%m-%d')), encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    return fh

def ndarray_to_string(array):
    """Converts the given ndarray into a unicode string.

    Parameters
    ----------
    array : ndarray

    Returns
    -------
    unicode
    """
    array = np.asarray(array)

    if array.ndim == 1:
        return u' '.join([unicode(item) for item in array])
    elif array.ndim == 2:
        return u'\n'.join([ndarray_to_string(line) for line in array])

    raise ValueError(u'only 1d arrays supported')


def save_object_to_file(obj, filename):
    """Saves the given object to file using cPickle.

    The object might provide extra routings for storing (e.g., __getstate__).

    Parameters
    ----------
    obj : any
        object to store
    filename : string
        file to store the object to
    """
    # Caution: using utf8_file_open doesn't work with cPickle
    cPickle.dump(obj, open(filename, "wb"), protocol=-1)

def sort_dict_by_key(d, reverse=False):
    """Sort the given dictionary by its keys.

    Parameters
    ----------
    d : dict
        dictionary to sort
    reverse : bool
        indicates if the sorting should be reversed

    Returns
    -------
    list of tupels
        contains tupels of key and value ordered according to key

    Examples
    --------
    >>> x = {'c':2, 'a':4, 'b':3, 'd':1, 'e':0}
    >>> sort_dict_by_key(x)
    [('a', 4), ('b', 3), ('c', 2), ('d', 1), ('e', 0)]

    >>> x = {'c':2, 'e':4, 'd':3, 'b':1, 'a':0}
    >>> sort_dict_by_key(x, True)
    [('e', 4), ('d', 3), ('c', 2), ('b', 1), ('a', 0)]
    """
    return sorted(d.iteritems(), key=itemgetter(0), reverse=reverse)

def sort_dict_by_label(d, reverse=False):
    """Sort the given dictionary by its values.

    Parameters
    ----------
    d : dict
        dictionary to sort
    reverse : bool
        indicates if the sorting should be reversed

    Returns
    -------
    list of tupels
        contains tupels of key and value ordered according to value

    Examples
    --------
    >>> x = {'c':2, 'a':4, 'b':3, 'd':1, 'e':0}
    >>> sort_dict_by_label(x)
    [('e', 0), ('d', 1), ('c', 2), ('b', 3), ('a', 4)]

    >>> x = {'c':2, 'e':4, 'd':3, 'b':1, 'a':0}
    >>> sort_dict_by_label(x, True)
    [('e', 4), ('d', 3), ('c', 2), ('b', 1), ('a', 0)]
    """
    return sorted(d.iteritems(), key=itemgetter(1), reverse=reverse)

def text_to_vocab_indices(vocab, tokens, unk=u'<UNK>'):
    """
    Convert all tokens in the text into their indices in the given vocabulary.

    Tokens that do not exist in the vocabulary will receive the <UNK> token
    index.

    Parameters
    ----------
    vocabulary : dict(str, int)
        mapping from token text to index
        must contain an UNKNOWN token
    tokens : str or list(str)
        text to replace all tokens in
    unk : str
        unknown word token

    Returns
    -------
    list(int)
        list that contains the vocabulary indices for all tokens instead of
        the tokens themselves
    list(str)
        list of the original input text having unknown tokens replaced by the
        unknown word token

    Examples
    >>> vocab = {u'i': 0, u'am': 1, u'home': 2, u'<UNK>':-1}
    >>> text_to_vocab_indices(vocab, u'i am home now .')
    ([0, 1, 2, -1, -1], [u'i', u'am', u'home', u'<UNK>', u'<UNK>'])
    >>> text_to_vocab_indices(vocab, [u'i', u'am', u'home', u'now', u'.'])
    ([0, 1, 2, -1, -1], [u'i', u'am', u'home', u'<UNK>', u'<UNK>'])
    """

    if isinstance(tokens, (str, unicode)):
        tokens = tokens.split()

    conv_tokens = [t if t in vocab else unk for t in tokens]
    sent_indices = [vocab[t] for t in conv_tokens]

    return sent_indices, conv_tokens

def utf8_file_open(filename, mode='r'):
    """Return a file object for the given filename in the given mode.

    Open an utf-8 file in the given mode (see io.open for further details) and
    uses only \n as line endings. Can open bz2 files.

    Parameters
    ----------
    filename : string
        name of the file to open
    mode : string
        open mode (see io.open for further details), default value: 'r'
    """

    # It seems that utf8 files are read properly by BZ2File.
    if filename.endswith(u'.bz2'):
        return codecs.getreader("utf-8")(BZ2File(filename, mode, compresslevel=9))

    return io.open(filename, mode, encoding='utf8', newline='\n')
