# Jelle Bosscher - UvA 2020
# jellebosscher@gmail.com

# Example script the tests all the IAT findings.
# Is able to replicate the WEAT if pos_op/POAT is set to False.

import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import clear_output
from nltk.corpus import wordnet as wn

#file imports
from pos_op import *
from pos_op_weat import *
from utils import *


embeddings_filename = "data/glove.840B.300d/glove.840B.300d.txt"
stimuli_folder = "data/IAT_target_attribute_lists/"

results = []
embeddings = load_embeddings(embeddings_filename)
data = construct_model(embeddings, stimuli_folder)

print("#"*75)
print("Starting experiments:\n")

for filename in os.listdir(stimuli_folder):
    input = []
    with open(stimuli_folder + filename, 'r') as f:
        for line in f:
            input.append(line.strip().split(', '))
    print("[{0}, {1}] vs [{2}, {3}]:".format(input[0][0], input[1][0],
                                       input[2][0], input[3][0]))
    X = input[0][1:]
    Y = input[1][1:]
    A = input[2][1:]
    B = input[3][1:]

    dist, test_stat, effect_size, pvalue = weat_iat_pos_op(data, X, Y, A, B)

    results.append([dist, test_stat, effect_size, pvalue])
