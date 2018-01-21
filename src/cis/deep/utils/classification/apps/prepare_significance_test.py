# -*- coding: utf-8 -*-
"""
"""

from argparse import ArgumentParser
from logging import getLogger
import sys

from cis.deep.utils import logger_config, file_line_generator, utf8_file_open

# import pydevd
# pydevd.settrace(host='129.187.148.250', stdoutToServer=True,
#         stderrToServer=True)

log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Prepare a predictions file created by classify.py for the
        use of Sebastian Padó's approximate randomization significance test.""")
parser.add_argument('prediction_file',
        help="""File containing a classifiers prediction created by classify.py
        .""")
parser.add_argument('outfile',
        help="""converted file""")

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('converting file')

    with utf8_file_open(args.outfile, 'w') as outfile:

        for line in file_line_generator(args.prediction_file):

            if line.startswith(u'#'):
                continue

            (_, true_label, pred_label) = line.split(';')
            true_label = int(true_label)
            pred_label = int(pred_label)

            tp = 1 if true_label == 1 and pred_label == 1 else 0
            model_pos = 1 if pred_label == 1 else 0
            gold_pos = 1 if true_label == 1 else 0

            outfile.write(u'%d %d %d\n' % (tp, model_pos, gold_pos))
    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
