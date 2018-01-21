#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""util.py: Collection of useful utilities."""

from itertools import islice, izip_longest
import re
import sys

from cis.deep.utils import file_line_generator


LOG_FORMAT = '%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s'

def Enum(**enums):
    """An enumeration factory class."""
    obj = type('Enum', (), enums)
    obj.named_value = dict([(a, v) for a, v in vars(obj).items() if not a.startswith('__')])
    obj.value_named = dict([(v, a) for a, v in obj.named_value.items()])
    return obj

def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type_, value, tb)
        print('\n')
        # ...then start the debugger in post-mortem mode.
        pdb.pm()

def extract_results_from_logfile(logfile, result='train_error', fmt='new',
        no_of_val_files=1):
    """Extract results from a given logfile and returns them as ndarray.

    Parameters
    ----------
    logfile : str
        path of the logfile
    result : str
        type of the result to be extracted; one of 'train_error',
        'val_error', 'val_ppl'
    format : str
        'new' or 'old', new format allows several validation files; old format
        only allowed 1 validation file.
    no_of_val_files : int
        number of validation files used in the logfile; is only matters if
        result = 'val_error' or 'val_perplexity'

    Returns
    -------
    ndarray
        contains all results in an array
    """

    if fmt == 'old':
        val_method_name = 'validate'
    else:
        val_method_name = '_validate_single_file'


    if result == 'train_error':
        pattern = re.compile(r'run\tAverage loss on .*? training set is (.*)',
                re.UNICODE)
    elif result == 'val_error':
        pattern = re.compile(
                r'%s\tAverage loss on .*? validation set is (.*)' % val_method_name,
                re.UNICODE)
    elif result == 'val_ppl':
        pattern = re.compile(
                r'%s\tPerplexity on .*? validation set is (.*)' % val_method_name,
                re.UNICODE)
    else:
        raise ValueError('Unknown result type to be extracted from logfile: %s'
                % result)

    values = list()

    for line in file_line_generator(logfile):
        match = re.search(pattern, line)

        if not match:
            continue

        values.append(float(match.group(1)))

    # Converts the 1d list of results into one list per validation file.
    if (result == 'val_error' or result == 'val_ppl') and no_of_val_files != 1:
        values = list(grouper_recipes(values, no_of_val_files))
        values = zip(*values)

    return values

def grouper(iterable, n):
    """Group n items from the iterable into a group.

    Parameters
    ----------
    iterable : any
        iterator to get the items from
    n : int
        number of items to form one group

    Returns
    -------
    tuple(items)
        tuple of n items taken from the iterator
    """
    chunk = tuple(islice(iterable, n))

    if not chunk:
        return
    yield chunk

def grouper_recipes(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks.
    Grouper taken from https://docs.python.org/2/library/itertools.html.
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def prepare_brown_signature(signature, max_size, add_right=False):
    """Convert variable length signatures into fixed length ones.

    Prepends zeros to the front of the signature.

    Parameters
    ----------
    signature : str
        brown signature a string (space separated)
    max_size : int
        size of the fixed signature
    add_right : bool
        indicates whether to add the padding zeros to the right of the signature
        instead of the left

    Returns
    -------
    str
        fixed length brown signature

    Example
    -------
    >>> prepare_brown_signature(u'1 1', 4)
    u'0 0 1 1'

    >>> prepare_brown_signature(u'1 1 1 1', 4)
    u'1 1 1 1'

    >>> prepare_brown_signature(u'1 1', 4, True)
    u'1 1 0 0'

    >>> prepare_brown_signature(u'1 1 1 1', 4, True)
    u'1 1 1 1'
    """
    sig_len = len(signature.split())
    needed_padding = max_size - sig_len

    if needed_padding == 0:
        return signature

    padding = u' '.join([u'0' for _ in xrange(needed_padding)])
    return padding + u' ' + signature \
        if not add_right else signature + u' ' + padding
