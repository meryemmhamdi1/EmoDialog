#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script uses trained pmi model to extend GALC lexicon based on semantic similarities
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "../Algorithms")
from DataPreProcessing import *
from nltk.corpus import stopwords

__author__ = "Meryem M'hamdi"
__email__ = "meryem.mhamdi@epfl.ch"

pmi_txt_file = 'C:/Users/fnac/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Deliverables/pmi_lexicon_new.txt'
pmi_dict_file = 'C:/Users/fnac/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Models/clean_pmi_updated.npy'
lexicon_file = "'C:/Users/fnac/Dropbox/meryem/algorithms/Data/ExtendedGALCLexicon.xls'"

pmi_dict = np.load(pmi_dict_file).item()

extended_galc_lexicon = pd.read_excel()
sm_list = list_nrc_lexicon(extended_galc_lexicon)

flat_lexicon_list = []
for i in range(0,len(sm_list)):
    for word in sm_list[i]:
        flat_lexicon_list.append(word)

distinct_words = []
for (word1,word2) in pmi_dict.keys():
   stop_words = list(set(stopwords.words('english')))
   if word1 not in stop_words and word2 not in stop_words:
       if word1 not in distinct_words and word1 not in flat_lexicon_list:
           distinct_words.append(word1)
       if word2 not in distinct_words and word2 not in flat_lexicon_list:
           distinct_words.append(word2)

print (len(distinct_words))
emotions = extended_galc_lexicon.columns.values


# Emotional Vectors of words using PMI
w, h = len(distinct_words), 20
matrix_word= [[0 for x in range(h)] for y in range(w)]

f = open(pmi_txt_file, 'w')
j = 0
for word in distinct_words:
    i = 0
    f.write(str(word)+'\t')
    for emotion in range(0, len(emotions)): # Iterate over all emotions => fill in the emotional vectors for all words
        for representative_word in sm_list[emotion]:
            r = len(sm_list[emotion])
            if (word,representative_word) in pmi_dict and (representative_word,word) not in pmi_dict:
                matrix_word[j][i] += pmi_dict.get((word, representative_word), 0) / r
            if (representative_word,word) in pmi_dict and (word,representative_word) not in pmi_dict:
                matrix_word[j][i] += pmi_dict.get((representative_word,word), 0) / r
            if (representative_word,word) in pmi_dict and (word,representative_word) in pmi_dict:
                matrix_word[j][i] += pmi_dict.get((representative_word,word), 0) / r
        f.write(str(matrix_word[j][i])+'\t')
        i +=1
    j +=1
    f.write('\n')

f.close()
