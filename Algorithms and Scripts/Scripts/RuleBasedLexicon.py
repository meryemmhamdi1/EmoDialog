#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script uses preprocessed dialogue feature set and applies rule based approach with no semantic extension
 to compute emotionalities and find the dominant emotion category for each subtitle
"""
import pandas as pd
import sys
sys.path.insert(0, "../Algorithms")
from DataPreProcessing import *
from SEMProjectSemanticModule import *
from SEMProjectSentenceModule import *
import ast

###### STEP 1: Loading Data with tokenized and affective representation:
# HERE YOU CAN CHANGE THE NAME OF THE FILE FROM WHICH TO LOAD THE DATA
number_emotions = 10
dialogues_df = pd.read_csv("/Users/MeryemMhamdi/Dropbox/meryem/Deliverables/3. Algorithms and Results" \
                  "/EmotionRecognition/RefinedAffectiveRepresentation/Unannotated_Representation_Whole.csv"
                  ,encoding='utf-8')

nava_repr = dialogues_df['Nava Representation']

# Convert nava_dialogues 
nava_dialogues = []
for i in range(0, len(nava_repr)):
    result = ast.literal_eval(nava_repr[i])
    nava_dialogues.append(result)


###### STEP 2: Loading Lexicon:

nrc_lexicon = "../../../1. Raw Dataset/Lexicons/LexiconNRC.xls"
galc_lexicon = '../../../1. Raw Dataset/Lexicons/ExtendedGALCLexicon.xls'
lexicon = pd.read_excel(nrc_lexicon)


###### STEP 3: Word Level
print "COMPUTING WORD LEVEL SEMANTICS:"
matrix_sentences_lexicon = compute_matrix_sentences_list_lexicon(nava_dialogues,lexicon,number_emotions)

print "Finished Computing Matrix Sentences"
###### STEP 4: Sentence Level:
 
# Emotion Recognition
print "COMPUTING SENTENCE LEVEL SEMANTICS:"
sentence_vectors_lexicon = compute_sentence_emotion_vectors_nrc(matrix_sentences_lexicon)

sentence_vectors_sent = compute_sentence_sentiment_vectors_nrc(matrix_sentences_lexicon)

emotionalities = compute_emotionalities_nrc(sentence_vectors_lexicon)

sentiments = compute_sentiments_nrc(sentence_vectors_sent,emotionalities)

###### FINAL STEP 5: Storing Emotion for each dialogues
print "STORING RESULTS: "

emo_dict_galc = {
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

emo_dict_nrc = {
    0: 'Anger',
    1: 'Anticipation',
    2: 'Disgust',
    3: 'Fear',
    4: 'Joy',
    5: 'Sadness',
    6: 'Surprise',
    7: 'Trust',
    8: 'Neutral'
}

sent_dict_nrc = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}
emotions = []
senti = []
for i in range(0,len(emotionalities)):
    emotions.append(emo_dict_galc[emotionalities[i]])
    senti.append(emo_dict_galc[sentiments[i]])

lexicon_results_df = pd.DataFrame()

lexicon_results_df['Nava dialogues'] = nava_dialogues

lexicon_results_df['Emotion'] = emotions

lexicon_results_df['Emotion ID'] = emotionalities

lexicon_results_df['Sentiment'] = senti

lexicon_results_df['Sentiment ID'] = sentiments

lexicon_results_df['Emotion Vectors'] = sentence_vectors_lexicon

lexicon_results_df['Sentiment Vectors'] = sentence_vectors_sent

lexicon_results_df.to_csv('/Users/MeryemMhamdi/Desktop/emotion_lexicon_vectors_nrc.csv')

