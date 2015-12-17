#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CHAILLOUX Cecile
# n° etu 21201448
# Projet HMM
# V1

from collections import defaultdict
import numpy as np

START = ('START', 'START')
STOP = ('STOP', 'STOP')


class Tagger:
    # Probabilités des transitions P(CATn | CATn-1, CATn-2)
    # Par exemple, self.transitions[ "V" ][("N", "DET")] doit contenir P( "V" | "N", "DET" )
    # Probabilités des emissions P( mot | CAT )
    # Par exemple, self.emissions["chat"]["N"] doit contenir P( "chat" | "N") !!!!
    #
    def __init__(self, order=2) :
        self.order = order
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))

    def words(self, sent, order=0):
        o = order or self.order

        s = [START]*o + sent + [STOP]
        wx = []
        for i, w in enumerate(s[o:]):
            prev = tuple(p[1] for p in s[i:i+o])
            wx.append((prev, w))

        return wx

    def train(self, corpus, smooth=1e-5):
        tag_freqs = defaultdict(float)
        prev_freqs = defaultdict(float)

        # comptage
        #
        for s in corpus:
            for prev, (w, c) in self.words(s):
                self.emissions[w][c] += 1
                self.transitions[prev][c] += 1
                tag_freqs[c] += 1
                prev_freqs[prev] += 1

        cats = tag_freqs.keys()

        # normalisation
        #
        for cat in cats:
            prevs = prev_freqs.keys()
            for prev in prevs:
                prev_normal = prev_freqs[prev] + len(cats) * smooth
                self.transitions[prev][cat] = (self.transitions[prev][cat] + smooth) / prev_normal
                print("P(%s|%s) = %f" % (cat, ",".join(prev), self.transitions[prev][cat]))

            words = self.emissions.keys()
            for word in words:
                if word == 'STOP': continue
                word_normal = tag_freqs[cat] + len(words) * smooth
                self.emissions[word][cat] = (self.emissions[word][cat] + smooth) / word_normal
                print("P(%s|%s) = %f" % (word, cat, self.emissions[word][cat]))


def main():
    tagger = Tagger(order=1)
    corpus = [[("Le", "DET"), ("chat", "N"), ("dort", "V")],
              [("Le", "DET"), ("chien", "N"), ("noir", "A"), ("mange", "V")],
              [("Un", "DET"), ("lapin", "N"), ("féroce", "A"), ("mange", "V"), ("des", "DET"), ("humains", "N")],
              [("Ma", "DET"), ("fille", "N"), ("attardée", "A"), ("est", "V"), ("une", "DET"), ("vraie", "A"), ("merveille", "N")],
              [("La", "DET"), ("reine", "N"), ("de", "PREP"), ("Narnia", "N"), ("est", "V"), ("horrible", "A")]]
    tagger.train(corpus)

if __name__ == '__main__':
    main()
