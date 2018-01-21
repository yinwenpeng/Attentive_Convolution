"""
Requires the enum34 package.
"""

from logging import getLogger

from enum import Enum, IntEnum

from cis.deep.utils import file_line_generator, logger_config, utf8_file_open,\
    sort_dict_by_label
import numpy as np


log = getLogger(__name__)
logger_config(log)

class SpecialToken(Enum):
    """Enum for special tokens and their string expression.

    Get the enum entry's value with SpecialToken.PAD.value.
    """
    UNKNOWN = u'<UNK>'
    SENT_START = u'<S>'
    SENT_END = u'</S>'
    PAD = u'<PAD>'


SPECIAL_TOKENS = [SpecialToken.UNKNOWN, SpecialToken.SENT_START,
        SpecialToken.SENT_END, SpecialToken.PAD]


class SpecialTokenID(IntEnum):
    """Enum for ids of special tokens.

    Get the enum entry's value with SpecialTokenId.PAD.value.
    """
    UNKNOWN = 0
    SENT_START = 1
    SENT_END = 2
    PAD = 3


def compute_avg_text_embedding(text, vocab, embs):
    """Convert the given text into a compressed vector using average embeddings.

    Average all word vectors to a final document vector.

    Parameters
    ----------
    text : str
        text to be compressed
    vocab: dict(str, int)
        vocabulary (see read_vocabulary_id_file)
    embs : ndarray(m*n)
        embeddings
    """
    vec = np.zeros(embs.shape[1])
    count = 0

    for tok in text.split():
        vec += embs[vocab.get(tok, SpecialTokenID.UNKNOWN.value), :]
        count += 1

    return vec / float(count)

def read_vocabulary_file(input_file, add_special_tokens=True):
    """Read the textual vocabulary into a list. Items that are empty after
    calling str.strip on them will be mapped to u'<EMPTY>'.

    Parameters
    ----------
    input_file : str
        location of the vocabulary
    add_special_tokens : bool
        indicates whether or not to add special tokens to the front of the
        vocabulary, like <UNK> for unknown tokens, etc.

    Returns
    -------
    list(str)
        vocabulary from token to unique id
    """
    vocab = list(file_line_generator(input_file))

    if add_special_tokens:
        _add_special_tokens(vocab)

    return [v.strip() if v.strip() else u'<EMPTY>' for v in vocab]

def read_vocabulary_id_file(input_file, add_special_tokens=True):
    """Read the textual vocabulary into a map that maps the token to it's index.

    Each map entry points from the vocabulary token to the index in the
    vocabulary.

    Parameters
    ----------
    input_file : str
        location of the vocabulary
    add_special_tokens : bool
        indicates whether or not to add special tokens to the front of the
        vocabulary, like <UNK> for unknown tokens, etc.

    Returns
    -------
    dict(str, int)
        vocabulary from token to unique id
    """
    vocab = read_vocabulary_file(input_file, add_special_tokens)
    vocab_to_indices = {w : i for (i, w) in enumerate(vocab)}

    if len(vocab) != len(vocab_to_indices):
        log.warning("""Vocabulary contains duplicate items. They have been
                removed automatically.""")
    return vocab_to_indices

def write_vocabulary_file(output_file, vocab):
    """Write the given vocabulary to the given file.

    The vocabulary items are stored in order of the vocab values, i.e., in the
    same order as they have been read by read_vocabulary_id_file.

    Parameters
    ----------
    output_file : str
        filename of the output
    vocab : dict(str, int)
        vocabulary that has been read by read_vocabulary_id_file
    """

    with utf8_file_open(output_file, 'w') as vocab_file:
        vocab_file.write(u'\n'.join(k[0]
                for k in sort_dict_by_label(vocab)))
        vocab_file.write(u'\n')

def _add_special_tokens(vocab):
    """Add special tokens to the beginning of the given vocabulary.

    Adds the special tokens only if they don't already exist. If the vocabulary
    already contains some special tokens the order of them does not change.

    Parameters
    ----------
    vocab : list(str)
        vocabulary items

    Returns
    -------
    list(str)
        vocabulary with the special tokens inserted at the front
    """
    if SpecialToken.PAD.value not in vocab:
        vocab.insert(0, SpecialToken.PAD.value)
    if SpecialToken.SENT_END.value not in vocab:
        vocab.insert(0, SpecialToken.SENT_END.value)
    if SpecialToken.SENT_START.value not in vocab:
        vocab.insert(0, SpecialToken.SENT_START.value)
    if SpecialToken.UNKNOWN.value not in vocab:
        vocab.insert(0, SpecialToken.UNKNOWN.value)

    return vocab
