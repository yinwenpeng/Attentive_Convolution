# -*- coding: utf-8 -*-
"""This file contains classes dealing with various corpora."""

from logging import getLogger
import re

from cis.deep.utils import logger_config, file_line_generator


log = getLogger(__name__)
logger_config(log)

class AmazonProductReviewCorpusReader:
    """Helper methods for the Amazon review corpus."""

    def __init__(self, infile):
        """
        Parameters
        ----------
        infile : str
        """
        self.infile = infile

    def review_generator(self, remove_meta_cols=True):
        """Iterate over all reviews

        Parameters
        ----------
        remove_meta_cols : bool
            indicates whether or not to remove the first 7 meta data columns
        """

        for line in file_line_generator(self.infile):
            line = line.decode(errors='ignore')

            if remove_meta_cols is True:
                line = self._extract_body(line)

            yield line

        raise StopIteration()

    @staticmethod
    def _extract_body(line):
        """Return the review body from the given text line."""
        # Remove the meta data in front of the line. Since the body may contain
        # more tabs, we need to join it here.
        body = u' '.join(line.split(u'\t')[7:])
        body = body.strip()
        # There are too many whitespaces.
#         body = re.sub(r'\s+', ' ', body, flags=re.UNICODE)
        return body
