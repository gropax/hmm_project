#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CHAILLOUX Cecile
# n° etu 21201448
# Projet HMM
# V1

from collections import defaultdict
# On importe tout ce qui est dans corpus

from .corpus import *

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
        self.matrice = defaultdict()

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
            for prev, (w, cat) in self.words(s):
                c = cat
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
        # En fait avec le corpus German Tiger c'est déjà une liste, pas besoin de faire de spilt
        """words = sentence.split()"""
        words = sentence

        list_tags_prev = []

        for i in range(self.order):
            words = ["START"] + words
            list_tags_prev.append("START")

        #print(words)

        for i, word in enumerate(words[self.order:]):
            #print(word)
            j = i + self.order
            prev_liste = []
            for indice in range(1, self.order + 1):
                blabla = j - indice

                prev_liste.append(list_tags_prev[j-indice])
            prev_inverse = tuple(prev_liste)
            prev = prev_inverse[::-1]

            max = 0
            cat_max = ""
            for tag in self.liste_cat:


                if self.emissions[word][tag] == 0.0:
                    self.emissions[word][tag] = float(1)/float(len(self.liste_cat))
                if self.transitions[prev][tag] == 0.0:
                    self.transitions[prev][tag] = float(1)/float((len(self.liste_cat)**self.order))
                """
                print("on a word = " + word + " et tag = " + tag + " et prev = " + str(prev))

                print(word)

                print("emission vaut : " + str(self.emissions[word][tag]))
                print("transition vaut : " + str(self.transitions[prev][tag]))
                """

                score = self.emissions[word][tag] * self.transitions[prev][tag]
                if score >= max:
                    max = score
                    cat_max = tag
            list_tags_prev.append(cat_max)
        return list_tags_prev


    def evalTagger(self, sentences_lst):
        acc = 0.0
        tot = 0.0
        for x,y in sentences_lst :
            y_hat = self.predict(x)
            tot += len(y_hat)
            for i,tag in enumerate(y_hat) :
                if tag == y[i] :
                    acc += 1
        return acc / tot


    def evalTagger_2(self, corpus_test):
        acc = 0.0
        tot = 0.0

        longueur = 0
        for i in corpus_test:
            liste_mot = []
            liste_cat = []
            
            for x in i:
                liste_mot.append(x[0])
                liste_cat.append(x[1])
                longueur += 1
            
            tot += len(liste_cat)
            liste_cats_predites = self.predict(liste_mot)
            liste_sans_start = liste_cats_predites[self.order:]
            
            self.confMatrix(liste_sans_start, liste_cat)
            """
            print("On devrait avoir :")
            print(liste_cat)

            print("On a :")
            print(liste_sans_start)
            """

            for j in range(len(liste_cat)):
                if liste_sans_start[j] == liste_cat[j]:
                    acc += 1
        return acc / tot



    def confMatrix(self, tag_pred, tag_list):
        for i in range(len(tag_pred)):
            if self.matrice.has_key((tag_pred[i], tag_list[i])):
                self.matrice[(tag_pred[i], tag_list[i])] += 1
            else:
                self.matrice[(tag_pred[i], tag_list[i])] = 1



def main():

    # L'hmm utilisé avec un corpus
    tagger3 = Tagger(order=3)
    tagger2 = Tagger(order=2)
    tagger1 = Tagger(order=1)
    
    tiger_train = ConllCorpus(TIGER_CORPUS)
    tiger_test = ConllCorpus(TIGER_TEST)
    smooth=1e-5
    
    #cat = "NN"
    #print(tagger.converter(tiger_univ_tags, cat))
    
    # On entraine l'hmm sur un corpus annoté
    tagger3.train(tiger_train)
    
    # On teste l'hmm sur un corpus et on compare les prédictions aux catégories réelles
    print("Avec order = 3")
    print(tagger3.evalTagger_2(tiger_test))
    print(tagger3.matrice)
    
    print("Avec order = 2")
    tagger2.train(tiger_train)
    print(tagger2.evalTagger_2(tiger_test))
    print(tagger2.matrice)
    
    print("Avec order = 1")
    tagger1.train(tiger_train)
    print(tagger1.evalTagger_2(tiger_test))
    print(tagger1.matrice)
    
    tagger.train(tiger_train)

    """
    print("La liste des cats est:")
    print(tagger.liste_cat)

    print("Le tableau des emissions est:")
    print(tagger.emissions)

    print("Le tableau des transitions est:")
    print(tagger.transitions)
    """

    
    tiger_test = ConllCorpus(TIGER_TEST)
    print(tagger.evalTagger_2(tiger_test))
    """
    sentences_lst = []
    for i in tiger_test:
        print(tagger.evalTagger_2(i))
    """
    #print(tagger.evalTagger(sentences_lst))
    """
    for i in tiger_test:
        print("Pour :")
        liste_mots = []
        liste_cat_reelles = []
        for mot in i:
            liste_mots.append(mot[0])
            liste_cat_reelles.append(mot[1])
        print(liste_cat_reelles)
        print("La prédiction est :")
        print(tagger.predict(liste_mots))

    """



    """
    # Petit corpus créé de toutes pièces pour tester l'HMM
    corpus_train = [[("Le", "DET"), ("chat", "N"), ("dort", "V")], [("Une", "DET"), ("chatte", "N"), ("boit", "V"), ("le", "DET"), ("lait", "N")], [("Mes", "DET"), ("souris", "N"), ("courent", "V")], [("Le", "DET"), ("chien", "N"), ("noir", "A"), ("mange", "V")], [("La", "DET"), ("chatte", "N"), ("grise", "A"), ("boit", "V"), ("son", "DET"), ("lait", "N")]]



    sentence1 = ("Le chat noir mange le chien noir")
    sentence2 = ("Des enfants nourrissent le chien")

    bon_result1 = ["START", "START", "START", "DET", "N", "A", "V", "DET", "N", "A"]
    bon_result2 = ["START", "START", "START", "DET", "N", "V", "DET", "N"]

    tagger.train(corpus_train)
    result1 = tagger.predict(sentence1)
    result2 = tagger.predict(sentence2)

    if result1 == bon_result1 :
        print("Phrase 1 correctement prédite")
    else:
        print("Phrase 1 mauvaise")
    if result2 == bon_result2 :
        print("Phrase 2 correctement prédite")
    else:
        print("Phrase 2 mauvaise")
    """
if __name__ == '__main__':
    main()


