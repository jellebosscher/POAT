# Jelle Bosscher - UvA 2020
# jellebosscher@gmail.com

# implementation of the POAT. Also implements the WEAT in python.

import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import clear_output
from nltk.corpus import wordnet as wn
import scipy

def error_matrix(A, B):
    """
    Calculates error matrix E by:
        - first diagonalizing B - A
        - construct E by setting positive entries of B - A to 0,
        and changing the sign of the negative eigenvalues.
    Input: A, B (NxN np.array, NxN np.array)
    Output: error matix E (NxN np.array)
    """
    B_min_A = (B - A)
    w, v = np.linalg.eig(B - A)
    np.minimum(w, 0, w)
    w *= -1.0
    return v@np.diag(w)@np.linalg.inv(v)


def K_E(A, B):
    """
    Compute the size of the error term as a proportion of the size of A.
    Input: positve operators A,B (NxN np.array, NxN np.array)
    Output:  Graded hyponymy [0,1] (float)
    """
    E = error_matrix(A,B)
    return 1 - np.linalg.norm(E)/np.linalg.norm(A)


def K_BA(A, B):
    """
    Compute the extent to which A is a hyponym of B by measuring the
    proportion of positive and negative eigenvalues.
    Input: positve operators A,B (NxN np.array, NxN np.array)
    Output:  Graded hyponymy [0,1] (float)
    """
    w, _ = np.linalg.eig(B - A)
    return np.sum(w)/np.sum(np.abs(w))


def weat_ass_K_E(tar, attr_set_1, attr_set_2, measure=K_E):
    """
    Measure association in terms of graded hyponymy or cosine_similarity (PAOT or WEAT).
    """
    if tar.shape != (300,300):
        sim_1 = np.mean([cosine_similarity(tar.reshape(1, -1), attr.reshape(1, -1)) for attr in attr_set_1])
        sim_2 = np.mean([cosine_similarity(tar.reshape(1, -1), attr.reshape(1, -1)) for attr in attr_set_2])
    else:
        sim_1 = np.mean([measure(tar, attr) for attr in attr_set_1])
        sim_2 = np.mean([measure(tar, attr) for attr in attr_set_2])

    return sim_1 - sim_2


def diff_weat_ass_K_E(model, tar_set_1, tar_set_2, attr_set_1, attr_set_2, measure=K_E):
    """
    Calculate the differential association between two target sets and two attribute sets.
    """
    pos_ops_tar_1 = [model[word] for word in tar_set_1]
    pos_ops_tar_2 = [model[word] for word in tar_set_2]
    pos_ops_attr_1 = [model[word] for word in attr_set_1]
    pos_ops_attr_2 = [model[word] for word in attr_set_2]

    association_1 = [weat_ass_K_E(tar, pos_ops_attr_1, pos_ops_attr_2, measure) for tar in pos_ops_tar_1]
    association_2 = [weat_ass_K_E(tar, pos_ops_attr_1, pos_ops_attr_2, measure) for tar in pos_ops_tar_2]

    mean_1 = np.mean(association_1)
    mean_2 = np.mean(association_2)

    test_stat = mean_1 - mean_2
    effect_size = weat_effect_size(test_stat, association_1, association_2)
    return test_stat, effect_size


def weat_effect_size(test_stat, association_1, association_2):
    """
    Calcuate the effect_size according to the WEAT.
    """
    assocs = association_1 + association_2
    mean = np.mean(association_1 + association_2)
    std_dev = (np.sum([(x - mean)**2 for x in assocs])/(len(assocs)-1))**0.5
    return (test_stat)/std_dev


def nullDistribution(model, tar1, tar2, att1, att2, iterations=100000, pos_op=True, measure=K_E):
    """
    Estimate the null distribution of the test statistics for N iterations of permutations of the target words.
    """
    all_targets = tar1 + tar2
    tar_length = len(all_targets)
    set_length = int(tar_length/2)
    att1_length, att2_length = len(att1), len(att2)

    distribution = np.zeros((iterations))
    idx = np.arange(tar_length)

    att1_null_matrix = np.zeros((att1_length, tar_length))
    att2_null_matrix = np.zeros((att2_length, tar_length))
    print("# of targets:", tar_length)
    print("null distribution for att_set_1\n# attributes:", att1_length, "\nprogress...", end='')

    for i, a in enumerate(att1):
        if i % 4 == 0:
            print(' ', i, end='')
        for j, t in enumerate(all_targets):
            if pos_op:
                att1_null_matrix[i][j] = measure(model[t], model[a])
            else:
                att1_null_matrix[i][j] = cosine_similarity(model[t].reshape(1, -1),
                                                           model[a].reshape(1, -1))
    print("---> done")
    print("null distribution for att_set_2\n# attributes:",att2_length,"\nprogress...", end='')

    for i, a in enumerate(att2):
        if i % 4 == 0:
            print(' ', i, end='')
        for j, t in enumerate(all_targets):
            if pos_op:
                att2_null_matrix[i][j] = measure(model[t], model[a])
            else:
                att2_null_matrix[i][j] = cosine_similarity(model[t].reshape(1, -1),
                                                           model[a].reshape(1, -1))
    print("---> done")

    print("-"*50)
    print("building distribution with test statistics...", end="")

    for dist_i in range(iterations):
        r_indices = np.random.permutation(idx)

        mean_tar1_att1 = np.mean(att1_null_matrix[:, r_indices[:set_length]])
        mean_tar1_att2 = np.mean(att2_null_matrix[:, r_indices[:set_length]])
        mean_tar2_att1 = np.mean(att1_null_matrix[:, r_indices[set_length:]])
        mean_tar2_att2 = np.mean(att2_null_matrix[:, r_indices[set_length:]])

        distribution[dist_i] = (mean_tar1_att1 - mean_tar1_att2) - mean_tar2_att1 + mean_tar2_att2

    print("---> done")
    return distribution


def calc_cumulative_prob(dist_values, x):
    """
    Calculate the cumulative probability of x.
    """
    d = scipy.stats.norm(np.mean(np.sort(dist_values)), np.std(dist_values, ddof=0))
    cdf = d.cdf(x)
    if cdf < 0.5:
        return cdf
    return  1 - cdf
