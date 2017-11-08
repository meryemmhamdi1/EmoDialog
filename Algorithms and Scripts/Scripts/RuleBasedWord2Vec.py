#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script uses preprocessed dialogue feature set and applies rule based word2vec approach to compute emotionalities
"""

import pandas as pd
import sys
sys.path.insert(0, "/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Algorithms")
from DataPreProcessing import *
from SEMProjectSemanticModule import *
from SEMProjectSentenceModule import *
from SentSyntacticModule import *
import ast
from gensim.models import word2vec

__author__ = "Meryem M'hamdi"
__email__ = "meryem.mhamdi@epfl.ch"

##### STEP 1: Loading Data with tokenized and affective representation:
# HERE YOU CAN CHANGE THE NAME OF THE FILE FROM WHICH TO LOAD THE DATA
print "LOADING DATA FILE"
dialogues_df = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Representation6.csv', encoding='utf-8')

tokenized_lemma = dialogues_df['Tokenized Lemmatized']

# Convert tokenized_lemma
tokenized_lemmatized_dialogues = []
for i in range(0, len(tokenized_lemma)):
    result = ast.literal_eval(tokenized_lemma[i])
    tokenized_lemmatized_dialogues.append(result)

nava_repr = dialogues_df['Nava Representation'] 

# Convert nava_dialogues 
nava_dialogues = []
for i in range(0, len(nava_repr)):
    result = ast.literal_eval(nava_repr[i])
    nava_dialogues.append(result)

###### STEP 2: Loading Lexicon:
print "LOADING LEXICON"
extended_galc_lexicon = pd.read_excel('/home/meryem/Dropbox/meryem/algorithms/Data/ExtendedGALCLexicon.xls')

###### STEP 3: Loading Word2Vec Model:
print "LOADING WORD2VEC MODEL"
model = word2vec.Word2Vec.load("/home/meryem/Desktop/EmotionRecognition/Semi-Supervised/300features_40minwords_10context_subtitle_new_one")

###### STEP 4: Word Level
print "COMPUTING WORD LEVEL SCORES"
matrix_sentences_word2vec = compute_matrix_sentences_list_word2vec(nava_dialogues,extended_galc_lexicon,model)


###### STEP 5: Sentence Level:
print "COMPUTING SENTENCE LEVEL SCORES"
# Emotion Recognition
sentence_vectors_word2vec = compute_sentence_emotion_vectors(matrix_sentences_word2vec)

emotionalities = compute_emotionalities(sentence_vectors_word2vec)

###### FINAL STEP 6: Storing Emotion for each dialogues

print "STORING EMOTIONS"
emo_dict= {
    0 :'Involvement-Interest', # +
    1 : 'Amusement-Laughter', # +
    2 : 'Pride-Elation', # +
    3 : 'Happiness-Joy',
    4 : 'Enjoyment-Pleasure',
    5 : 'Tenderness-Feeling Love',
    6 : 'Wonderment-Feeling Awe',
    7 : 'Feeling Disburdened- Relief',
    8 : 'Astonishment- Surprise',
    9 : 'Longing- Nostalgia',
    10 : 'Pity-Compassion',
    11 : 'Sadness-Despair',
    12 : 'Worry-Fear',
    13 : 'Embarrassement-Shame',
    14 : 'Guilt-Remorse',
    15 : 'Disappointment- Regret',
    16 : 'Envy-Jealousy',
    17 : 'Disgust-Repulsion',
    18 : 'Contempt-Scorn',
    19 : 'Irritation-Anger',
    20 : 'Neutral'
}

emotions = []
for i in range(0,len(emotionalities)):
    emotions.append(emo_dict[emotionalities[i]])

word2vec_results_df = pd.DataFrame()

word2vec_results_df['Nava dialogues'] = nava_dialogues

word2vec_results_df['Emotion'] = emotions

word2vec_results_df.to_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Word2VecBased/dialogues_Labelled_Word2Vec6.csv')

