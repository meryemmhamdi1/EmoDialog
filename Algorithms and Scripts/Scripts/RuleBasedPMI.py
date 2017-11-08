#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script uses preprocessed dialogue feature set and applies rule based pmi approach to compute emotionalities
"""

import pandas as pd
import sys
sys.path.insert(0, "../Algorithms")
from DataPreProcessing import *
from SEMProjectSemanticModule import *
from SEMProjectSentenceModule import *
import ast
import numpy as np
from tqdm import tqdm

__author__ = "Meryem M'hamdi"
__email__ = "meryem.mhamdi@epfl.ch"

##### STEP 1: Loading Data with tokenized and affective representation:
input_path_file = "/Users/MeryemMhamdi/Dropbox/meryem/Deliverables/3. Algorithms and Results" \
                  "/EmotionRecognition/RefinedAffectiveRepresentation/Unannotated_Representation_Whole.csv" # <----- HERE YOU CAN CHANGE THE NAME OF THE FILE FROM WHICH TO LOAD THE DATA
lexicon_path = '../../../1. Raw Dataset/ExtendedGALCLexicon.xls'  # <---- REPLACE HERE BY THE EXCEL FILE OF THE LEXICON
pmi_path_file = "/Users/MeryemMhamdi/Dropbox/meryem/Deliverables/3. Algorithms and Results" \
                "/EmotionRecognition/Models/PMI/ppmi_dict.npy" # <---- REPLACE HERE BY THE PATH OF THE PMI MODEL
output_path_file = "/Users/MeryemMhamdi/Dropbox/meryem/Deliverables/3. Algorithms and Results" \
                   "/EmotionRecognition/Results/PMIBased/Annotated_ppmi_new.csv" # <---- REPLACE HERE BY THE PATH OF THE OUTPUT FILE TO SAVE THE dialogues ANNOTATED WITH EMOTIONS

number_emotions = 20  # <--- CHANGE HERE THE NUMBER OF EMOTIONS

print ("LOADING DATA FILE")
dialogues_df = pd.read_csv(input_path_file,encoding='utf-8')

tokenized_lemma = dialogues_df['Tokenized Lemmatized']

# Convert tokenized_lemma
tokenized_lemmatized_dialogues = []
for i in tqdm(range(0, len(tokenized_lemma))):
    result = ast.literal_eval(tokenized_lemma[i])
    tokenized_lemmatized_dialogues.append(result)

nava_repr = dialogues_df['Nava Representation']

# Convert nava_dialogues
nava_dialogues = []
for i in tqdm(range(0, len(nava_repr))):
    result = ast.literal_eval(nava_repr[i])
    nava_dialogues.append(result)

###### STEP 2: Loading Lexicon:
print ("LOADING LEXICON")
extended_galc_lexicon = pd.read_excel(lexicon_path)

###### STEP 3: Loading Pre-trained PMI Dictionary:
print ("LOADING PMI DICTIONARY")
pmi_dict = np.load(pmi_path_file).item()

###### STEP 4: Word Level
print ("COMPUTING WORD LEVEL SCORES")
matrix_sentences_word_pmi = compute_matrix_sentences_list(nava_dialogues,extended_galc_lexicon,pmi_dict,number_emotions)


###### STEP 5: Sentence Level:
print ("COMPUTING SENTENCE LEVEL SCORES")
# Emotion Recognition
sentence_vectors_pmi = compute_sentence_emotion_vectors(matrix_sentences_word_pmi,number_emotions)

emotionalities = compute_emotionalities(sentence_vectors_pmi,number_emotions)

###### FINAL STEP 6: Storing Emotion for each dialogues

print ("STORING EMOTIONS")

# <--- DO NOT DELETE THIS PIECE JUST COMMENT IT OUT AND DEFINE THE LIST OF EMOTIONS TO ADHERE WITH YOUR MODEL
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

# <---
emotions = []
for i in tqdm(range(0,len(emotionalities))):
    emotions.append(emo_dict[emotionalities[i]])

pmi_results_df = pd.DataFrame()

pmi_results_df['Nava dialogues'] = nava_dialogues

pmi_results_df['Emotion'] = emotions

pmi_results_df['Emotion ID'] = emotionalities

pmi_results_df['Emotion Vectors'] = sentence_vectors_pmi 

pmi_results_df.to_csv(output_path_file)

