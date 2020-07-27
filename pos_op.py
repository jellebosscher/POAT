# Jelle Bosscher - UvA 2020
# jellebosscher@gmail.com

# Positive operator implementation

import numpy as np
from nltk.corpus import wordnet as wn

def pos_op(model, word, pos=None, depth=-1):
    """
    Builds a positive operator for a given word. Returns a NxN matrix, where N is the
    dimensions of the embedding used, which consists of the sum of outerproducts of its
    hyponyms. Case sensitive, will ommit hyponym gathering with proper nouns.
    Input: word (str)
    Output: NxN matrix (np.array)
    """
    n = len(next(iter(model.values())))
    output_matrix = np.zeros((n, n))

    if word[0].isupper(): #proper noun
        return np.outer(model[word], model[word])

    closure_set = get_hyponyms(word, pos, depth)

    found = 0

    for token in set(closure_set):
        try:
            vec = model[token]
            output_matrix = np.add(output_matrix, np.outer(vec,vec))
            found += 1
        except:
            pass

    if found == 0:
        print(word, " - not found", end="")
        if word not in model.keys():
            print("and in keys:", word in model.keys())
            return None
        print()
        return np.outer(model[word], model[word])

    return output_matrix

def get_hyponyms(word, pos=None, depth=-1):
    """
    Takes a word as input and return the transitive hyponymy closure according to wordnet.
    Assumes first entries are the correct ones.
    Input: word (str), depth (int, -1 means no limit)
    Ouput: list of words [str, ..., str]
    """
    hyponyms = []
    hypo = lambda s: s.hyponyms()

    for synset in wn.synsets(word, pos=pos):
        closure_syns = list(synset.closure(hypo, depth=depth)) # find transative clusure of synset

        closure_syns.append(synset) # include current synset
        for syn in closure_syns:
            for ln in syn.lemma_names():
                hyponyms.append(ln.lower())
    return hyponyms
