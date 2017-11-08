#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This scripts applies all NLP preprocessing steps to the dialogue dataset
   Returning the dataframe with a list of NAVA words for each subtitle text
"""

import pandas as pd
import sys
sys.path.insert(0, "../Algorithms")
from DataPreProcessing import *
from SEMProjectSemanticModule import *
from SEMProjectSentenceModule import *
from SentSyntacticModule import *
from tqdm import tqdm

__author__ = "Meryem M'hamdi"
__email__ = "meryem.mhamdi@epfl.ch"

input_file_path = "/Users/MeryemMhamdi/Dropbox/meryem/Deliverables/2. Aggregated Processed Dataset/Aggregated_Dialogues.csv" # <--- CHANGE HERE THE PATH OF THE INPUT FILE
output_file_path = "/Users/MeryemMhamdi/Dropbox/meryem/Deliverables/2. Aggregated Processed Dataset/PreProcessed_and_NAVA_Version.csv" # <--- CHANGE HERE THE PATH OF THE OUTPUT FILE

###### STEP 1: Loading Data:
# HERE YOU CAN CHANGE THE NAME OF THE FILE FROM WHICH TO LOAD THE DATA
dialogues = pd.read_csv(input_file_path, encoding ="utf-8")

dialogues['text'] = dialogues['Sentence'] # <--- CHANGE HERE THE NAME OF THE COLUMN IN THE CSV FILE

###### STEP 2: Replacing Special Categories:
print "SOME CLEANING >>>>>"
replaced_categories = handle_special_categories(dialogues)

###### STEP 3: Replacing contractions (needed for more accurate tokenization)
dialogues_no_contractions = replace_contractions(replaced_categories)

###### STEP 4: Tokenization of dialogues into words
print "BAG OF WORD REPRESENTATION >>>>>"
tokenized_list = bag_of_word_representation(dialogues_no_contractions)

###### STEP 5: Part of Speech Tagging:
print "Part of Speech Tagging >>>>>"
tagged_dialogues = pos_tagging(tokenized_list)

###### STEP 6: Tokenized Lemmatized Representation:
print "Lemmatization >>>>>"
new_tagged = normalize_pos_tags_words(tagged_dialogues)
tokenized_lemma = lemmatizer_raw(new_tagged)

print "LOADING spaCy NLP >>>>>"
###### STEP 7: Loading spaCy:
nlp = spacy.load('en')

print "Dependency Parsing >>>>>"
###### STEP 8: Dependency Parsing:
docs = []
# Joining text:
dialogues_text = []
for i in tqdm(range(0, len(tokenized_list))):
    space = u" "
    dialogues_text.append(space.join(tokenized_list[i]))
dialogues_text[0].encode("utf-8")
for i in tqdm(range(0, len(dialogues_text))):
    doc = nlp(dialogues_text[i])
    docs.append(doc)

new_samples = []
for sample in docs:
    new_samples_sub = []
    for word in sample:
        new_samples_sub.append((unicode(word.lemma_),word.pos_))
    new_samples.append(new_samples_sub)

print "Applying Syntactic Rules >>>>>>"

###### STEP 9: Applying Syntactic Rules:
new_samples,triple_dependencies = apply_syntactic_rules(docs,new_samples)

print "Further Cleaning >>>>>>>>>>"
###### STEP 10: Applying Named Entity Tagging:
dialogues_without_ne = remove_named_entities(new_samples)


###### STEP 11: Normalizing POS tag:
normalized_tags = normalize_pos_tags_words1(dialogues_without_ne)

###### STEP 12: Removal of Punctuation and Stop words and Converting to Lower Case and Removal of Other special categories: url, number, username:
tagged_dialogues_without = eliminate_stop_words_punct(normalized_tags)
print tagged_dialogues_without[0]
###### STEP 13: Lemmatization:
lemmatized_dialogues = lemmatizer(tagged_dialogues_without)
print lemmatized_dialogues[0]
lemmatized_dialogues_untag = lemmatizer_untagged(tagged_dialogues_without)
print lemmatized_dialogues_untag[0]
###### STEP 14: Keeping only NAVA words:
nava_dialogues = keep_only_nava_words(lemmatized_dialogues)
print nava_dialogues[0]
print "Storing in DataFrame"
###### STEP 16: Storing Tokenized Lemmatized + Affective Representation + Emotion for each dialogues
dialogues_df = pd.DataFrame()

dialogues_df['Tokenized Lemmatized'] = tokenized_lemma

dialogues_df['Nava Representation'] = nava_dialogues

dialogues_df.to_csv(output_file_path)
