# Jelle Bosscher - UvA 2020
# jellebosscher@gmail.com

# utils for POAT/WEAT
from pos_op_weat import *
from pos_op import *

def weat_iat_pos_op(data, X, Y, A, B, iterations=100000, POAT=True, measure=K_E):
    """
    Runs the POAT (or WEAT) on one experiment.
    Input:  data -    POAT=True: positive operators (NxN np.array),
                      POAT=False: word vectors (Nx1 np.array)
            X, Y, A, B - two sets of target words and two sets of attribute words.
            iterations - amount of iterations to estimate the likelihood of the effect_size
            POAT=True - use positive operators (POAT) or word embeddings (WEAT),
            measure  - applies to positive operators only, define measure of graded hyponymy
    """
    dist = nullDistribution(data, X, Y, A, B, iterations, POAT, measure)
    test_stat, effect_size = diff_weat_ass_K_E(data, X, Y, A, B, measure)
    pvalue = calc_cumulative_prob(dist, test_stat)

    print("#{}".format("-"*75))
    print(f"{'|':>37} effect_size: {effect_size}\r| test_stat: {test_stat}")
    print(f"{'|':>37} exponent: {np.floor(np.log10(np.abs(pvalue))+1.)}\r| pvalue: {pvalue}")
    print("#{}".format("-"*75))
    print()
    return dist, test_stat, effect_size, pvalue

def load_embeddings(filename):
    """
    Import embddings from a text file line by line. Store the vector for each words in a dictionary.
    """
    embeddings = dict()
    print("Loading embeddings...")
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word == '.':
                continue
            try:
                vector = np.asarray(values[1:], "float32")
            except ValueError:
                pass
            embeddings[word] = vector
    print("Done\n")
    return embeddings

def construct_model(embeddings, directory, POAT=True):
    """
    Read all words from every experiment file and build the model using only those words.
    Input: embeddings, directory, POAT (dict, path_to_folder, boolean (POAT vs WEAT))
    """
    input_words = []
    for filename in os.listdir(directory):
        with open(directory + filename, 'r') as f:
            for line in f:
                input_words.append(line.strip().split(', '))
    all_words = set([x for test in input_words for x in test[1:]])

    print("Construction model: pos_op or vec...")

    data = dict()

    for word in all_words:
        if POAT:
            data[word] = pos_op(embeddings, word)
        else:
            data[word] = embedding[word]

    print("Done\n")
    return data
