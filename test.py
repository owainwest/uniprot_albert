#!TODO: map output to 3 bins for each of these - make bins based on percentile given length k runs
# use itertools.combinations
# from  itertools import combinations

import pandas as pd
import json
from itertools import combinations
import numpy as np 
import statistics

def get_hydrophobicity(peptide, aa_features):
    k = len(peptide)
    
    values = [feats["hydrophobicity"] for feats in aa_features.values()]
    kmer_values = [sum(k_values) for k_values in combinations(values, k)]

    lower_bound = np.percentile(kmer_values, 33.33)
    upper_bound = np.percentile(kmer_values, 66.67)  
    print(lower_bound, upper_bound)
    DEFAULT_GUESS = statistics.median(kmer_values)

    res = 0
    for amino_acid in peptide:
        if amino_acid in aa_features:
            res += aa_features[amino_acid]["hydrophobicity"]
        else:
            res += DEFAULT_GUESS
            
    if res < lower_bound:
        return 0
    elif res < upper_bound:
        return 1
    else:
        return 2

with open(F"./aa_features.json", "r") as aa_feature_file:
    aa_feature_text = aa_feature_file.read()
aa_features = json.loads(aa_feature_text)


print(get_hydrophobicity("agu", aa_features))
