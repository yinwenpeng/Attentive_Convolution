# -*- coding: utf-8 -*-
"""
example usage:
-k 94
X:\sa\experiments\contextual_polarity\mlp\sent_1\amazon\brown\5000-0.1l-200\features-predict-out-unique
X:\sa\experiments\contextual_polarity\mlp\sent_1\amazon\brown\5000-0.1l-200\features-predict-clusters

-s X:\sa\experiments\contextual_polarity\mlp\sent_1\amazon\brown\5000-0.1l-200\features-predict-out-unique
X:\sa\experiments\contextual_polarity\mlp\sent_1\amazon\brown\5000-0.1l-200\features-predict-out-unique
X:\sa\experiments\contextual_polarity\mlp\sent_1\amazon\brown\5000-0.1l-200\features-predict-clusters
"""
from argparse import ArgumentParser
from logging import getLogger
import sys

from sklearn.cluster.k_means_ import KMeans

from cis.deep.utils import logger_config, utf8_file_open, save_object_to_file
import numpy as np


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(description="""Cluster given data points using
        k-means.""")

parser.add_argument('data_points', help='data points to be clustered')
parser.add_argument('outfile', help='output file')

parser.add_argument('-m', '--model', help='save model into that file')
parser.add_argument('-c', '--centroids', help='save centroids into that file')
parser.add_argument('-i', '--max-iterations', dest='max_iter', type=int,
        default=300, help='Maximum number of iterations of the algorithm')
parser.add_argument('-mr', '--root', action='store_true',
        help="""modify the data by taking the root of every entry before 
        clustering""")
parser.add_argument('-t', '--threads', type=int, default=1,
        help="""number of jobs using for the clustering""")

cluster_group = parser.add_mutually_exclusive_group(required=True)
cluster_group.add_argument('-k', '--clusters', type=int,
        help='number of clusters; either -k or -s must be given')
cluster_group.add_argument('-s', '--start-points', dest='start_points',
        help="""file that contains the start points for all clusters; either -k
        or -s must be given""")

def get_initial_centers(cluster_count, filename):
    """Return number of clusters and initial cluster centers or the method to
    create them.

    Parameters
    ----------
    cluster_count : None/int
        number of clusters; if None, loads the cluster centroids from the given
        file
    filename : None/str
        name of file, which contains the cluster centroids; if None,
        cluster_count must be given

    Returns
    -------
    if cluster_count is given: (int, str)
        cluster count and the method that will be used to choose the centroids
        later
    if cluster_count is not given (int, ndarray)
        cluster count and the centroids
    """

    if cluster_count:
        return (cluster_count, 'k-means++')

    centers = np.loadtxt(filename)
    return (centers.shape[1], centers)

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')
    data = np.loadtxt(args.data_points)

    if args.root is not None:
        data = np.sqrt(data)

    (k, initial_points) = get_initial_centers(args.clusters, args.start_points)

    log.info('calculate center points')
    kmeans = KMeans(k, initial_points, 1, args.max_iter, copy_x=False)
    predict = kmeans.fit_predict(data)

    log.info('storing results')

    if args.model:
        save_object_to_file(kmeans, args.model)

    with utf8_file_open(args.outfile, 'w') as outfile:

        for i in xrange(predict.shape[0]):
            outfile.write(u'%d\n' % predict[i])

    if args.centroids:
        np.savetxt(args.centroids, kmeans.cluster_centers_)

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
