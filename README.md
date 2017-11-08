# Overview:

This repository has code for Fine Grained Emotion Recognition applied to Human Dialogues. This semester project was completed in Fall 2016 in EPFL under the supervision of Professor Dr. Pearl Pu.

# Datasets and Lexicon:
We web scraped two TV series (The Big Bang Theory and Friends) in addition to Cornell Movie Dialog Corpus. 
Big Bang Theory subtitles were scraped from https://bigbangtrans.wordpress.com/ while Friends was web scraped from https://fangj.github.io/friends. 
Cornell Corpus was downloaded in text format from https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html.

The set of tweets used for the semi-supervised machine learning experiments is 2 million fine grained olympic tweets based annotated based on hashtags (http://www.aclweb.org/anthology/W13-1603).

The lexicon used in all experiments is the GALC 20 emotions the extended version  (http://hci.epfl.ch/sharing-emotion-lexicons-and-data/)

# Used Libraries

spaCy NLP Tools

Stanford Dependency Parser

gensim word2vec

NLTK

