"""
This class is used for finding groups of sharp features.
"""

# Take in list
# Change into ndarray
# Normalize

import numpy as np

from sklearn.cluster import DBSCAN
from multiprocessing.pool import Pool
from sklearn.preprocessing import StandardScaler


def dbscan(data):
    # data should be a list of tuples for positions of peaks (ex. [(2, 5, 3), (7, 4, 5), ... ]
    data = np.array(data)
    data = StandardScaler().fit_transform(data)

    db, eps, samps = make_label_plot(data)
    return db


def make_label_plot(data, num_workers=8):
    e_bounds = (0.01, 1)
    e_step = 0.1
    samp_bounds = (1, 10)
    samp_step = 1

    # Choose when least amount of change in label num
    e_range = range(e_bounds[0], e_bounds[1], step=e_step)
    samp_range = range(samp_bounds[0], samp_bounds[1], step=samp_step)

    proc = Pool(processes=num_workers)

    # Build arguments
    args = []
    for e in e_range:
        for s in samp_range:
            args.append((data, e, s))

    results = proc.starmap(scan, args)
    results = np.array(results).reshape([len(e_range), len(samp_range)])

    grads = np.gradient(results)

    abs_grad = np.zeros(grads[0].shape)
    for g in grads:
        abs_grad += abs(g)
    index = np.argmin(abs_grad)
    i = index // abs_grad.shape[0]
    j = index % abs_grad.shape[0]

    best_eps = list(e_range)[i]
    best_samp = list(samp_range)[i]

    return DBSCAN(eps=best_eps, min_samples=best_samp).fit(data), best_eps, best_samp


def scan(data, eps, samp):
    return DBSCAN(eps=eps, min_samples=samp).fit(data).labels_

