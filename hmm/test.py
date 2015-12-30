#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CHAILLOUX Cecile
# n° etu 21201448
# Projet HMM
# V1

from collections import defaultdict
import numpy as np

START = ('START', 'START')


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
        self.liste_cat = defaultdict()

    def words(self, sent, order=0):
        o = order or self.order

        s = [START]*o + sent
        wx = []
        for i, w in enumerate(s[o:]):
            prev = tuple(p[1] for p in s[i:i+o])
            wx.append((prev, w))

        return wx

# entraine sur un corpus pour remplir les tableaux emissions et transitions
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

        self.liste_cat = tag_freqs.keys()
        

        # normalisation
        #
        for cat in self.liste_cat:
            prevs = prev_freqs.keys()
            for prev in prevs:
                prev_normal = prev_freqs[prev] + len(self.liste_cat) * smooth
                self.transitions[prev][cat] = (self.transitions[prev][cat] + smooth) / prev_normal
                #print("P(%s|%s) = %f" % (cat, ",".join(prev), self.transitions[prev][cat]))

            words = self.emissions.keys()
            for word in words:
                word_normal = tag_freqs[cat] + len(words) * smooth
                self.emissions[word][cat] = (self.emissions[word][cat] + smooth) / word_normal
                #print("P(%s|%s) = %f" % (word, cat, self.emissions[word][cat]))


# utilise les tableaux remplis dans train pour prédire les catégories des mots de sentence
    def predict(self, sentence):
    
        #print(self.transitions)
    
        #print("hello")
        words = sentence.split()
        list_tags_prev = []
        for i in range(self.order):
            words = ["START"] + words
            list_tags_prev.append("START")
        print(words)
        print(list_tags_prev)
        
        
        for i, word in enumerate(words[self.order:]):
            j = i + self.order
            prev_liste = []
            for indice in range(1, self.order + 1):
                blabla = j - indice
                #print(blabla)
                #print(list_tags_prev[blabla])
                prev_liste.append(list_tags_prev[j-indice])
            prev_inverse = tuple(prev_liste)
            prev = prev_inverse[::-1]
            print
            print(prev)
            max = 0
            cat_max = ""
            for tag in self.liste_cat:
                
                
                if self.emissions[word][tag] == 0.0:
                    self.emissions[word][tag] = float(1)/float(len(self.liste_cat))
                if self.transitions[prev][tag] == 0.0:
                    self.transitions[prev][tag] = float(1)/float((len(self.liste_cat)**self.order))

                print("on a word = " + word + " et tag = " + tag + " et prev = " + str(prev))
                print("emission vaut : " + str(self.emissions[word][tag]))
                print("transition vaut : " + str(self.transitions[prev][tag]))


                score = self.emissions[word][tag] * self.transitions[prev][tag]
                if score >= max:
                    print("cette cat est + probable que les précédentes")
                    max = score
                    cat_max = tag
            list_tags_prev.append(cat_max)
        return list_tags_prev

		
    def evalTagger(self, sentences_lst):
        #print("hello")
        acc = 0.0
        tot = 0.0
        for x,y in sentences_lst :
            y_hat = self.predict(x)
            tot += len(y_hat)
            for i,tag in enumerate(y_hat) :
                if tag == y[i] :
                    acc += 1
        return acc / tot

		

def main():
    tagger = Tagger(order=3)
    corpus_train = [[("Le", "DET"), ("chat", "N"), ("dort", "V")], [("Une", "DET"), ("chatte", "N"), ("boit", "V"), ("le", "DET"), ("lait", "N")], [("Mes", "DET"), ("souris", "N"), ("courent", "V")], [("Le", "DET"), ("chien", "N"), ("noir", "A"), ("mange", "V")], [("La", "DET"), ("chatte", "N"), ("grise", "A"), ("boit", "V"), ("son", "DET"), ("lait", "N")]]
    
    sentence1 = ("Le chat noir mange le chien noir")
    sentence2 = ("Des enfants nourrissent le chien")
    
    bon_result1 = ["START", "START", "START", "DET", "N", "A", "V", "DET", "N", "A"]
    bon_result2 = ["START", "START", "START", "DET", "N", "V", "DET", "N"]
    
    tagger.train(corpus_train)
    result1 = tagger.predict(sentence1)
    result2 = tagger.predict(sentence2)

    print
    if result1 == bon_result1 :
        print("Phrase 1 correctement prédite")
    if result2 == bon_result2 :
        print("Phrase 2 correctement prédite")

if __name__ == '__main__':
    main()
	
	