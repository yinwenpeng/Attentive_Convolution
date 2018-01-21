# -*- coding: utf-8 -*-
"""
"""

from _collections import defaultdict
from argparse import ArgumentParser
from collections import Counter
from logging import getLogger
import sys

from cis.deep.utils import file_line_generator, logger_config, utf8_file_open, \
    sort_dict_by_key


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""Labels given clusters according to their majority class.
        """)
parser.add_argument('data_file',
        help="""contains a line for each example which consists of the example's
        original label and its cluster id separated by a space""")
parser.add_argument('predicted_labels',
        help="""output file containing the predicted labels for each item; one
        label per line""")
parser.add_argument('-cl', '--cluster_labels',
        help="""output file containing the mapping of cluster ids to new labels
        """)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')
    items = []

    for line in file_line_generator(args.data_file):
        items.append(tuple(line.split()))

    log.info('compute majority labels')
    cluster_to_label_count = defaultdict(Counter)

    # Count labels per cluster
    for (label, cluster_id) in items:
        cluster_to_label_count[cluster_id][label] += 1

    majority_labels = dict()

    # Get majority label per cluster
    for cluster_id in cluster_to_label_count:
        majority_labels[cluster_id] = cluster_to_label_count[cluster_id].most_common(1)[0][0]

    log.info('assign labels to examples')

    with utf8_file_open(args.predicted_labels, 'w') as pred_file:

        for example_line in file_line_generator(args.data_file):
            pred_file.write(majority_labels[example_line.split()[1]] + u'\n')


    if args.cluster_labels:

        with utf8_file_open(args.cluster_labels, 'w') as outfile:

            for (cluster_id, label) in sort_dict_by_key(majority_labels):
                outfile.write(u'%s %s\n' % (cluster_id, label))

    log.info('finished')


if __name__ == "__main__":
    sys.exit(main())
