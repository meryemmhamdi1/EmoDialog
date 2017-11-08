#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file has functions for Computing Word Level Emotionality
    using different Methodologies for Comparison:
     * PMI
     * Word2Vec
    It starts by training PMI and importing word2vec model and based on the similarities between the representative
     words of each emotion category and the word in question, it computes emotionality of the word
"""
from __future__ import division
from nltk.collocations import *
from tqdm import tqdm

__author__ = "Meryem M'hamdi"
__email__ = "meryem.mhamdi@epfl.ch"

def calculate_pmi(flatten_list_nava, unique_lexicon):
    """
    Trains a PMI model from scratch and forms a pmi dictionary (key is the word pair (w1,w2) where
    w1 belongs to the dialogue feature set and w2 is in the set of representative word for each emotion in the lexicon
     based on cooccurence matrix
    :param flatten_list_nava:
    :param unique_lexicon:
    :return:
    """
    finder = BigramCollocationFinder.from_words(flatten_list_nava, window_size=10)

    " Without Filter: use pmi matrix"
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    pmi = finder.score_ngrams(bigram_measures.pmi)

    " With Filter: use pmi1 matrix"
    finder.apply_freq_filter(13)
    pmi1 = finder.score_ngrams(bigram_measures.pmi)

    " Clean Dictionary: keeping only values for which the second the word is part of galc lexicon"
    clean_pmi = []
    for ((w1, w2), value) in pmi:
        if value < 0:
            value = 0
        if w2 in unique_lexicon:
            clean_pmi.append(((w1, w2), value))
        if w1 in unique_lexicon:
            clean_pmi.append(((w2, w1), value))

    clean_pmi_dict = dict(clean_pmi)
    return clean_pmi_dict

def compute_matrix_sentences_list_lexicon(nava_tweets, lexicon, number_emotions):
    """
    Applies
    :param nava_tweets:
    :param lexicon:
    :param number_emotions:
    :return:
    """
    matrix_sentence_whole = []
    for i in tqdm(range(0,len(nava_tweets))): # for each sentence
        w, h = len(nava_tweets[i]),number_emotions
        matrix_sentence = [[0 for x in range(w)] for y in range(h)]
        j = 0 
        for word in nava_tweets[i]: # for each word
            # Looking for match between that keyword and representative word in each emotion category in the lexicon
            for e in range(0,lexicon.shape[1]):
                if word in list(lexicon[e]):
                    matrix_sentence[e][j] = 1
            j += 1
        matrix_sentence_whole.append(matrix_sentence)
    return matrix_sentence_whole
    
def compute_matrix_sentences_list(transcript_words, nrc_lexicon, clean_pmi_dict,number_emotions):
    """

    :param clean_pmi_dict:
    :param transcript_words: we can pass any version of the bag of words
    :param galc_lexicon:
    :return:
    """

    sm_list = list_nrc_lexicon(nrc_lexicon)
    emotions = nrc_lexicon.columns.values
    matrix_sentences_list = []
    for i in tqdm(range(0, len(transcript_words))): # Iterate over all sentences
        " Initialize matrix for each sentence "
        w, h = len(transcript_words[i]), number_emotions
        matrix_sentence = [[0 for x in range(w)] for y in range(h)]
        k = 0
        for word in transcript_words[i]: # Iterate over all words in the sentence
            j = 0
            for emotion in range(0, len(emotions)): # Iterate over all emotions => fill in the emotional vectors for all words
                total_pmi = 0
                for representative_word in sm_list[emotion]:
                    r = len(sm_list[emotion])
                    total_pmi += clean_pmi_dict.get((word, representative_word), 0)
                if word in sm_list[emotion]:
                    matrix_sentence[j][k] += 10
                else:
                    matrix_sentence[j][k] += total_pmi / r  #np.power(total_pmi,1/r)
                j += 1 # increment index of representative words
            k += 1 # increment index of tweet words
        # append the matrix_sentence to the global list for all sentences
        matrix_sentences_list.append(matrix_sentence)
    return matrix_sentences_list

    
def compute_matrix_sentences_list_word2vec(transcript_words, galc_lexicon,model,number_emotions):
    """

    :param clean_pmi_dict:
    :param transcript_words: we can pass any version of the bag of words
    :param galc_lexicon:
    :return:
    """

    sm_list = list_nrc_lexicon(galc_lexicon)
    emotions = galc_lexicon.columns.values
    matrix_sentences_list = []
    for i in tqdm(range(0, len(transcript_words))): # Iterate over all sentences
        " Initialize matrix for each sentence "
        w, h = len(transcript_words[i]), number_emotions
        matrix_sentence = [[0 for x in range(w)] for y in range(h)]
        k = 0
        for word in transcript_words[i]: # Iterate over all words in the sentence
            j = 0
            for emotion in range(0, len(emotions)): # Iterate over all emotions => fill in the emotional vectors for all words
                total_similarity = 0
                for representative_word in sm_list[emotion]:
                    r = len(sm_list[emotion])
                    if word in model and representative_word in model:
                        total_similarity += model.similarity(word, representative_word)
                if word in sm_list[emotion]:
                    matrix_sentence[j][k] += 10
                else:
                    matrix_sentence[j][k] += total_similarity / r  #np.power(total_similarity,1/r)
                j += 1 # increment index of representative words
            k += 1 # increment index of transcript words
        # append the matrix_sentence to the global list for all sentences
        matrix_sentences_list.append(matrix_sentence)
    return matrix_sentences_list

