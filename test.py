#!TODO: map output to 3 bins for each of these - make bins based on percentile given length k runs
# use itertools.combinations
# from  itertools import combinations

import pandas as pd
import json
from itertools import combinations
import numpy as np 
import statistics

def get_pks(peptide, aa_features):
    k = len(peptide)
    values = [feats["pks"] for feats in aa_features.values()]

    kmers = list(combinations(values, k))
    kmer_values = [sum(sum(aa) for aa in kmer) for kmer in kmers]
    lower_bound = np.percentile(kmer_values, 33.33)
    upper_bound = np.percentile(kmer_values, 66.67)  
    # print(lower_bound, upper_bound)
    DEFAULT_GUESS = statistics.median(kmer_values)
    print("-*" * 50)
    print("Peptide: ", peptide)
    print(lower_bound, DEFAULT_GUESS, upper_bound)

    res = 0
    for amino_acid in peptide:
        if amino_acid in aa_features:
            res += sum(aa_features[amino_acid]["pks"])
        else:
            res += DEFAULT_GUESS
    print(">Res: ", res)
    if res < lower_bound:
        return 0
    elif res < upper_bound:
        return 1
    else:
        return 2

with open("./aa_features.json", "r") as aa_feature_file:
    aa_feature_text = aa_feature_file.read()
aa_features = json.loads(aa_feature_text)

test_list = ["agu", "nlp", "gut", "masha"]
results = [(elem, get_pks(elem, aa_features)) for elem in test_list]
print(results)
