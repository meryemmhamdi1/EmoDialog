#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file has some pre-processing utilities functions used for
    * Data cleaning
    * Tokenization
    * POS Tagging
    * NAVA word extraction
    * Named Entity Recognition
    * Lemmatization
    * Stop Word Removal
    * Lexicon Processing
"""


from tqdm import tqdm
import re
import io
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer


__author__ = "Meryem M'hamdi"
__email__ = "meryem.mhamdi@epfl.ch"


def handle_special_categories(dialogue_df):
    """
    Handling Entities/ Special categories:
    *   Replacing @ instances with <username>
    *   Replacing urls with <url>
    *   Replacing numbers/phone/fax with <number>
    :param dialogue_df:
    :return:
    """
    dialogue_list = []
    for i in tqdm(range(0, len(dialogue_df))):
        new_text = re.sub(r"http\S+", "", dialogue_df.iloc[i]['text'])
        new_text = re.sub(r"@\S+", "", new_text)
        new_text = re.sub(r"\d+", "", new_text)
        new_text = re.sub(r"#", "", new_text)
        dialogue_list.append(new_text)
    dialogue_df['text'] = dialogue_list
    return dialogue_df

def replace_contractions(dialogue_df):
    """
    Replaces contractions like "I don't" to "I do not" for more accurate tokenization and results
    :param dialogue_df:
    :return:
    """
    f = io.open('../Algorithms/contractions.txt', 'r',
                encoding='utf8')
    text = f.read()
    contractions = eval(text)
    keys = list(contractions.keys())
    values = list(contractions.values())
    for i in range(0, len(contractions)):
        dialogue_df = dialogue_df.replace({keys[i]: values[i]}, regex=True)
    return dialogue_df

def is_ascii(s):
    """
    Detecting Ascii characters and removing non Ascii characters
    :param s:
    :return:
    """
    return all (ord(c) < 128 for c in s)

def bag_of_word_representation(dialogue_df):
    """
    Tokenization, UTF-8 decoding and Removal of white spaces
    :param dialogue_df:
    :return: set of bag of words
    """
    dialogues_bag_words = []
    tokenizer = RegexpTokenizer(r'\w+')
    for dialogue in dialogue_df['text']:

        # Removing of non-ascii character
        non_ascii_dialogue = re.sub(r'[^\x00-\x7F]+','',dialogue)

        # Tokenization
        dialogues_tokenized = [t for t in tokenizer.tokenize(non_ascii_dialogue)]
        dialogues_bag_words.append(dialogues_tokenized)
    return dialogues_bag_words

def pos_tagging(dialogues_bag_words):
    """
    POS tagging of dialogues using universal tagset
    :param dialogues_bag_words:
    :return:
    """
    tagged_dialogues = []
    for i in tqdm(range(0, len(dialogues_bag_words))):
        tagged_dialogues.append(nltk.pos_tag(dialogues_bag_words[i]))
    return tagged_dialogues

def normalize_pos_tags_words(tagged_dialogues):
    """
    Categorizing Penn Tags to Noun, Verb, Adjective, Adverb for easy extraction of NAVA words
    :param tagged_dialogues:
    :return: dialogues_nava
    """
    dialogues_nava = []
    dialogues_nava_sub = []
    for i in range(0, len(tagged_dialogues)):
        for (word, tag) in tagged_dialogues[i]:
            if tag == 'NN' or tag == 'NNP' or tag == 'NNPS' or tag == 'NNS':
                dialogues_nava_sub.append((word, 'n'))
            elif tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
                dialogues_nava_sub.append((word, 'v'))
            elif tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
                dialogues_nava_sub.append((word, 'Adj'))
            elif tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                dialogues_nava_sub.append((word, 'Adv'))
            else:
                dialogues_nava_sub.append((word, tag))
        dialogues_nava.append(list(dialogues_nava_sub))
        dialogues_nava_sub = []
    return dialogues_nava


def keep_only_nava_words(tagged_dialogues):
    """
    :param tagged_dialogues:
    :return: dialogues_nava
    """
    dialogues_nava = []
    dialogues_nava_sub = []
    for i in range(0, len(tagged_dialogues)):
        for (word, tag) in tagged_dialogues[i]:
            if tag == "n" or tag == "v" or tag =="ADJ" or tag == "ADV":
                dialogues_nava_sub.append(word)
        dialogues_nava.append(list(dialogues_nava_sub))
        dialogues_nava_sub = []
    return dialogues_nava

def extract_entity_names(tree):
    """
    Extracting Named Entities by working through the NER tag tree
    :param tree:
    :return:
    """
    non_entity_names = []
    entity_names = []
   
    if hasattr(tree, 'label') and tree.label:
        if tree.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in tree]))
        else:
            for child in tree:
                entity_names.extend(extract_entity_names(child))
    else:
        non_entity_names.append(tree)
    return non_entity_names

def remove_named_entities(dialogue_df):
    """
    Removing Named Entities such as time, names of place, proper nouns, etc and so on because of their objectivity
    is not relevant for Emotion Recognition Task
    :param new_samples:
    :return:
    """
    dialogues_without_ne = []
    for dialogues in dialogue_df:
        nre_dialogues = nltk.ne_chunk(dialogues, binary = True)
        non_entity_names = []
        for tree in nre_dialogues:    
            non_entity_names.extend(extract_entity_names(tree))
        dialogues_without_ne.append(non_entity_names)
    return dialogues_without_ne


def lemmatizer(dialogues_tagged):
    """
    Lemmatizing the dialogue text using Word Net Lemmatizer using pos tags information
    :param dialogues: tagged text
    :return: lemmatized dialogue
    """
    dialogues_lemmatized = []
    lmtzr = WordNetLemmatizer()
    for i in range(0,len(dialogues_tagged)):
        dialogues_sub = []
        for (word,tag) in dialogues_tagged[i]:
            if tag=='v' or tag =='n':
                dialogues_sub.append((lmtzr.lemmatize(word,tag),tag))
            else: 
                dialogues_sub.append((word,tag))
        dialogues_lemmatized.append(dialogues_sub)
    return dialogues_lemmatized



def eliminate_stop_words_punct(tagged_dialogues):
    """
    Elimination of Stop words using the English list
    Elimination of Punctuation

    :rtype: object
    :param tagged_dialogues:
    :return: tagged_dialogues_without
    """
    stop_words = list(set(stopwords.words('english')))
    non_emotinal_verbs = ['go','be','do','have','get']
    customized_stop_words = stop_words + non_emotinal_verbs
    tagged_dialogues_without = []
    for i in range(0, len(tagged_dialogues)):
        tagged_dialogues_without_sub = []
        for (word, tag) in tagged_dialogues[i]:
            if word not in customized_stop_words and word not in ['url','number','username'] and len(word) >= 2:
                tagged_dialogues_without_sub.append((word.lower(), tag))
        tagged_dialogues_without.append(tagged_dialogues_without_sub)
    return tagged_dialogues_without

def make_unique(duplicate_list):
    """
    Returns the distinct list of vocabulary of the dialogue dataset
    :param duplicate_list:
    :return:
    """
    unique_words = list(set(duplicate_list))
    return unique_words

def make_unique_lexicon(lexicon):
    """
    Processes lexicon dataframe to extract the distinct list of representative words of the lexicon
    :param lexicon:
    :return: unique_lexicon
    """
    lexicon_flatten = []
    emotions = lexicon.columns.values
    for i in range(0, len(emotions)):
        for representative_word in lexicon[emotions[i]].dropna():
            lexicon_flatten.append(representative_word)
    unique_lexicon = make_unique(lexicon_flatten)
    return unique_lexicon

def extract_list_lexicon(lexicon):
    """
    Extracts the list of representative words for each emotion category in the lexicon
    :param lexicon:
    :return: sm_list
    """
    emotions = lexicon.columns.values
    sm_list = []
    for emotion in emotions:
        sm = list(lexicon[emotion].dropna())
        sm_list.append(sm)
    return sm_list
