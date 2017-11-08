#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script trains PMI dictionary
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "../Algorithms")
from DataPreProcessing import *
from SEMProjectSemanticModule import *
import ast
from tqdm import tqdm

__author__ = "Meryem M'hamdi"
__email__ = "meryem.mhamdi@epfl.ch"

input_path_file = "/Users/MeryemMhamdi/Dropbox/meryem/Deliverables/2. Aggregated Processed Dataset/PreProcessed_and_NAVA_Version.csv" # <--- REPLACE HERE BY THE RESULT OF RUNNING SCRIPT AffectiveRepresentation.py
output_path_file = "/Users/MeryemMhamdi/Dropbox/meryem/Deliverables/3. Algorithms and Results/EmotionRecognition/Models/PMI/ppmi_dict.npy" # <---- REPLACE HERE BY THE PATH OF THE OUTPUT FILE WHERE TO SAVE THE PMI MODEL
lexicon_path = '../../../1. Raw Dataset/ExtendedGALCLexicon.xls'  # <---- REPLACE HERE BY THE EXCEL FILE OF THE LEXICON

print ("Preparing Dataset")

whole_df = pd.read_csv(input_path_file,encoding='utf-8')
tokenized_lemma = whole_df['Tokenized Lemmatized']

# Convert tokenized_lemma
tokenized_lemmatized_tweets = []
for i in tqdm(range(0, len(tokenized_lemma))):
    result = ast.literal_eval(tokenized_lemma[i])
    tokenized_lemmatized_tweets.append(result)


flatten_list = [word for sublist in tokenized_lemmatized_tweets for word in sublist]

print ("Extracting Lexicon")
extended_galc_lexicon = pd.read_excel(lexicon_path)
unique_lexicon = make_unique_lexicon(extended_galc_lexicon)


print ("Training PMI dictionary")
pmi_dict = calculate_pmi(flatten_list, unique_lexicon)

print ("Saving PMI dictionary")
np.save(output_path_file, pmi_dict)

# <--- PRINTING TO CHECK THE VALUES  COULD BE NEEDED FOR UNDERSTANDING THE RANGE OF VALUES AND DOING PROPER SCALING
pmi_dict_read = np.load(output_path_file).item()
print (pmi_dict_read)

# Checking an already trained pmi model
pmi_dict = np.load('C:/Users/fnac/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Models/clean_pmi.npy').item()

for (word1,word2) in pmi_dict.keys():
    if word1 == 'awesome' or word2== 'good':
        print "("+ str(word1) +", "+ str(word2) + ") = " + str(pmi_dict[(word1,word2)])

