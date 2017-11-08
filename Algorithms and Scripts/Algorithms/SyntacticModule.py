#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file has function that handcods rules for treating certain syntactic dependencies:
   * Negation dependency (in verbs, adjectives, nouns)
   * Adjectival complement
   * Adjectival modifier
   * Adverbial modifier

"""
from spacy.symbols import nsubj, advmod, acomp, amod, neg, NOUN, VERB, ADJ, ADV

__author__ = "Meryem M'hamdi"
__email__ = "meryem.mhamdi@epfl.ch"

def apply_syntactic_rules(docs,new_samples):
    """
    Given the syntactic dependency tree, extract the dependencies of interest and apply rules to treat them
    :param docs:
    :param new_samples:
    :return:
    """
    triple_dependencies = []
    samples = docs
    for i in range(0,len(samples)):
        negated_verbs = []
        negated_nouns = []
        triple_dependencies_sub = []
        for j in range(0,len(samples[i])):
            word = samples[i][j]
            triple_dependencies_sub.append((word.lemma_,word.dep_,word.head.lemma_))
            # Looking for negation dependency
            if word.dep == neg:
                new_samples[i].append(("not_"+str(word.head.lemma_),word.head.pos_))
                new_samples[i] = [x for x in new_samples[i] if x != (unicode(word.head.lemma_),word.head.pos_)
                                  and x!=(unicode(word.lemma_),word.pos_)]
                if word.head.pos == VERB:
                    negated_verbs.append(word.head.i)
            # Looking for adjectival complement
            if word.dep == acomp:
                if word.head.i in negated_verbs:
                    new_samples[i].append(("not_"+str(word.lemma_),word.pos_))
                    new_samples[i] = [x for x in new_samples[i] if x != ("not_"+str(word.head.lemma_),word.head.pos_)
                                      and x != (unicode(word.lemma_),word.pos_)]
                else: 
                    new_samples[i] = [x for x in new_samples[i] if x != (unicode(word.head.lemma_),word.head.pos_)]
            # Looking for negation in nouns and adjectives
            if word.lemma_ == "no" or word.lemma_ == 'not' or word.lemma_ == "never":
                if word.head.pos == NOUN:
                    new_samples[i].append(("not_"+str(word.head.lemma_),word.head.pos_))
                    new_samples[i] = [x for x in new_samples[i] if x != (unicode(word.head.lemma_),word.head.pos_)
                                      and x!=(unicode(word.lemma_),word.pos_)]
                    negated_nouns.append(word.head.i)
            # Looking for adjectival modifier 
            if word.dep == amod:
                if word.head.i in negated_nouns:
                    new_samples[i].append(("not_"+str(word.lemma_),word.pos_))
                    new_samples[i] = [x for x in new_samples[i] if x != ("not_"+str(word.head.lemma_),word.head.pos_)
                                      and x != (unicode(word.lemma_),word.pos_)]
                else:
                    new_samples[i] = [x for x in new_samples[i] if x != (unicode(word.head.lemma_),word.head.pos_)]
            # Looking for adverbial modifier
            if word.dep == advmod:
                new_samples[i] = [x for x in new_samples[i] if x != (unicode(word.lemma_),word.pos_)]
        triple_dependencies.append(triple_dependencies_sub)
    return new_samples,triple_dependencies