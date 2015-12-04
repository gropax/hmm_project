#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CHAILLOUX Cecile
# n° etu 21201448
# Projet HMM
# V1

from __future__ import print_function
from codecs import open
from collections import defaultdict

import numpy as np

class HiddenMarkovModel():

    def __init__(self) :
        
        # Probabilités des transitions P(CATn | CATn-1, CATn-2)
        # Par exemple, self.transitions[ "V" ][("N", "DET")] doit contenir P( "V" | "N", "DET" )
        self.transitions = defaultdict(lambda : defaultdict(float))
        
        # Probabilités des emissions P( mot | CAT )
        # Par exemple, self.emissions["chat"]["N"] doit contenir P( "chat" | "N") !!!!
        self.emissions = defaultdict(lambda : defaultdict(float))
        self.smooth = 1e-7



    def train(self, corpus):
        
        
        # comptage
        tag_freqs = defaultdict(float)
        tuple_prev_freqs = defaultdict(float)
        for s in corpus:
            s = [("", "START"), ("", "START")] + s + [("", "STOP")]
        
            for i, (w,c) in enumerate(s[2:]):
                j = i + 2
                tuple_prev = (s[j-2][1], s[j-1][1])
                
                self.emissions[w][c] += 1
                self.transitions[tuple_prev][c] += 1
                
                tag_freqs[c] += 1
                tuple_prev_freqs[tuple_prev] += 1
                
                
                print(i, w, c, tuple_prev)

        # normalisation des transitions
        for cat in self.emissions.keys():
            for tuple_prev in tuple_prev_freqs.keys():
                self.transitions[tuple_prev][cat] =  np.log((self.transitions[tuple_prev][cat] + self.smooth) / tuple_prev_freqs[tuple_prev])
                print(" apres " + str(tuple_prev) + " la proba d avoir " + str(cat) +  " est = " + str(self.transitions[tuple_prev][cat]))


        # normalisation des émissions
        for word in self.emissions.keys():
            for cat in tag_freqs.keys():
                self.emissions[word][cat] = np.log((self.emissions[word][cat] + self.smooth) / tag_freqs[cat])
                print(" avec " + str(cat) + " la proba d avoir " + str(word) +  " est = " + str(self.emissions[word][cat]))



def main():
    tagger = HiddenMarkovModel()
    corpus = [[("Le", "DET"), ("chat", "N"), ("dort", "V")], [("Le", "DET"), ("chien", "N"), ("noir", "A"), ("mange", "V")]]
    
    tagger.train(corpus)



main()