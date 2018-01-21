from _collections import defaultdict
from collections import Counter

def purity(clusters, classes):
    """Compute purity for the given data.

    Parameters
    ----------
    clusters : list(int)
        cluster ids of all examples
    classes : list(int)
        class ids of all examples
    """

    d = defaultdict(list)

    # Get a list of class numbers of all examples in a cluster.
    for k, v in zip(clusters, classes):
        d[k].append(v)

    mayority = 0

    # Count the mayority class number and add it up over all clusters.
    for k in d:
        mayority += Counter(d[k]).most_common(1)[0][1]

    return float(mayority) / len(clusters)
