import numpy as np
import pandas as pd
import pickle
import copy
import math

class TFIDF:
    def __init__(self, dataset):
        self.data = dataset
        self.data_size = len(dataset)
        self.tf = [{} for _ in range(self.data_size)]
        self.df = {}
        self.tf_idf = None

    def solve(self):
        for i in range(self.data_size):
            words = self.data[i].split(' ')
            word_set = set()
            for word in words:
                word_set.add(word)
                if word not in self.tf[i]:
                    self.tf[i][word] = 1.0
                else:
                    self.tf[i][word] += 1.0
            for word in word_set:
                if word not in self.df:
                    self.df[word] = 1.0
                else:
                    self.df[word] += 1.0
        self.tf_idf = copy.deepcopy(self.tf)
        for words in self.tf_idf:
            for word in words:
                words[word] *= math.log(self.data_size/self.df[word])


if __name__ == "__main__":
    with open('data/sentence_of_T.txt') as f:
        tfidf = TFIDF(f.readlines())
