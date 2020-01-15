
"""
Purpose: Bin packing sample optimization
Date created: 2020-01-10

Contributor(s):
    Mark M.
"""


import numpy as np
import pandas as pd


space_per_bin = 9

def bin_sample(ascending=False):
    widths = [2,3,4,5,6,7,8]
    qty = [4,2,6,6,2,2,2,]
    tmp_matrix = [[widths[r] for c in range(qty[r])] for r in range(len(widths))]
    if ascending:
        return [x for y in tmp_matrix for x in y]
    else:
        return sorted([x for y in tmp_matrix for x in y], reverse=True)


def iter_bin_sample():
    widths = [2,3,4,5,6,7,8]
    qty = [4,2,6,6,2,2,2,]
    tmp_matrix = [[widths[r] for c in range(qty[r])] for r in range(len(widths))]
    for i in tmp_matrix:
        for j in i:
            yield j


sample_list = bin_sample(False)

def first_fit_decreasing(value_list, bins):
    slack = [bins]
    sol = [[]]
    for itm in value_list:
        for j, bin_ in enumerate(slack):
            if bin_ >= itm:
                slack[j] -= itm
                sol[j].append(itm)
                break
        else:
            sol.append([itm])
            slack.append(bins - itm)
    return sol

res = first_fit_decreasing(sample_list, space_per_bin)


ibs = iter_bin_sample()
i_sample_list = [i for i in ibs]


def bubble_sort(seq):
    """
    Inefficiently sort the mutable sequence (list) in place.

    seq MUST BE A MUTABLE SEQUENCE.

    As with list.sort() and random.shuffle this does NOT return 
    """
    index_list = np.arange(len(seq) - 1)
    changed = True
    while changed:
        changed = False
        for i in index_list:
            if seq[i] > seq[i+1]:
                seq[i], seq[i+1] = seq[i+1], seq[i]
                changed = True
    return seq
 
bubble_sort(sample_list)




































