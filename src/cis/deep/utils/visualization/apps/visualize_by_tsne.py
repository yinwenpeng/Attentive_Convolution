# -*- coding: utf-8 -*-
"""
The required t-sne implementation used can be found at
https://github.com/turian/textSNE

example usage:
X:/sa/embeddings/ColWes08/combined-WilWieHof05
-f X:/sa/embeddings/ColWes08/combined-non_WilWieHof05-shuffle-1500

X:/sa/embeddings/vlbl/general-nce5-1ep/vLBL_1_14-03-05_07-09-05.embeddings_r-combined-WilWieHof05
-f X:/sa/embeddings/vlbl/general-nce5-1ep/vLBL_1_14-03-05_07-09-05.embeddings_r-combined-non_WilWieHof05
"""

from argparse import ArgumentParser
from logging import getLogger
import sys

from calc_tsne import tsne
from cis.deep.utils import file_line_generator, logger_config
from cis.deep.utils.visualization import render_points
import numpy as np
import pylab as plt


log = getLogger(__name__)
logger_config(log)

parser = ArgumentParser(
        description="""This script creates a 2d visualization of different kinds
        of input, e.g., Collobert & Weston word embeddings or RAE query
        representations. The code is a modification of a original t-SNE code.
        """)
parser.add_argument('file', type=str, help='first file to load')
parser.add_argument('-f', '--file2', type=str, help='second file to load')
parser.add_argument('-o', '--out', type=str,
        help='write the rendered image to the given output file')

def scaleData(x):
    """Scales the given data between 0 and 1.
    This is necessary, because t-sne will fail for too big numbers.
    """
    x -= np.min(x)
    x /= np.max(x)
    return x

def getData(emb_file):
    """Load the data file.

    Parameters
    ----------
    emb_file : str
        name of the data file in which the first tab-separated column contains
        the title and the second column the values of an item

    Returns
    -------
    list(str)
        item titles
    list(ndarray)
        item values
    """
    titles = []
    data = []

    for l in file_line_generator(emb_file):
        token, emb = l.split(u'\t')
        titles.append(token)
        data.append(np.fromstring(emb, sep=u' '))

    return titles, np.asarray(data)

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    log.info('start parameters: ' + str(args))

    log.info('loading data')
    titles, x = getData(args.file)

    file_size1 = len(titles)

    if args.file2 is not None:
        titles2, x2 = getData(args.file2)
        titles.extend(titles2)
        x = np.vstack((x, x2))

#     x = scaleData(x)

    log.info('performing t-SNE')
    out = tsne(x, no_dims=2, perplexity=30, initial_dims=100, use_pca=False)

    points = [('green', [(title, point[0], point[1])
            for title, point in zip(titles[:file_size1], out[:file_size1, :])])]

    if args.file2 is not None:
        points.append(('gray', [(title, point[0], point[1])
            for title, point in zip(titles[file_size1:], out[file_size1:, :])]))

#     pca = PCA(n_components=2)
#     out = pca.fit_transform(x)

#     mds = MDS()
#     out = mds.fit_transform(x)

    log.info('rendering result')
    render_points(points, 20, 20)

    if args.out:
        plt.savefig(args.out, dpi=600)
    else:
        plt.show()

    log.info('finished')

if __name__ == "__main__":
    sys.exit(main())
