#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
TIGER_CORPUS = os.path.join(DATA_DIR, 'german_tiger_train.conll')
TIGER_TEST = os.path.join(DATA_DIR, 'german_tiger_test.conll')
TIGER_UNIV_TAGS = os.path.join(DATA_DIR, 'de_tiger.map')


class ConllCorpus:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, 'r') as corpus:
            sent = []
            for l in corpus:
                line = l.strip()
                if line:
                    fields = line.split('\t')
                    word, tag = fields[1], fields[3]
                    sent.append((word, tag))
                else:
                    yield sent
                    sent = []


class TigerCorpus(ConllCorpus):
    def __init__(self, corpus):
        super().__init__(corpus)


class UnivTigerCorpus(TigerCorpus):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.fetch_tags()

    def __iter__(self):
        for sent in super().__iter__():
            yield [(w, self.tags[t]) for w, t in sent]

    def fetch_tags(self):
        self.tags = {}
        for line in open(TIGER_UNIV_TAGS, 'r'):
            fine, univ = line.strip().split('\t')
            self.tags[fine] = univ
