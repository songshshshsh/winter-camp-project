import numpy as np
import pandas as pd
import pickle
import copy
import math
import sys

class TFIDF:
    def __init__(self, dataset):
        self.data = dataset
        self.data_size = len(dataset)
        self.tf = [{} for _ in range(self.data_size)]
        self.norm = []
        self.df = {}
        self.tf_idf = None
        self.solve()

    def solve(self):
        for i in range(self.data_size):
            if i % 100000 == 0:
                print("\tprocess: {}/{}".format(i, self.data_size))
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
            tmp_norm = 0.0
            for word in words:
                words[word] *= math.log(self.data_size/self.df[word])
                tmp_norm += words[word] * words[word]
            self.norm.append(np.sqrt(tmp_norm))
    
    def make_pair(self, index1, index2):
        best_pair = []
        for i in range(len(index1)):
            if i % 100 == 0:
                print("\tprocess: {}/{}".format(i, len(index1)))
            select = -1
            max_sim = -1
            dict_t = self.tf_idf[index1[i]]
            for j in range(len(index2)):
                dict_f = self.tf_idf[index2[j]]
                if len(dict_t) < len(dict_f):
                    sim = self.cal_sim(index1[i], index2[j])
                else:
                    sim = self.cal_sim(index2[j], index1[i])
                if sim > max_sim:
                    select = j
                    max_sim= sim
            best_pair.append([index1[i], index2[select], max_sim])
        return best_pair

    def cal_sim(self, index1, index2):
        dict1 = self.tf_idf[index1]
        dict2 = self.tf_idf[index2]
        sim = 0.0
        for word in dict1:
            if word not in dict2:
                continue
            sim += dict1[word] * dict2[word]
        return sim/self.norm[index1]/self.norm[index2]


def solve(from_type, to_type, tfidf, labels):
    from_index, to_index = [], []
    data_size = len(labels)
    for i in range(data_size):
        label = labels[i].rstrip('\n')
        if len(tfidf.tf_idf[i]) > 20:
            continue
        if label == from_type:
            from_index.append(i)
        elif label == to_type:
            to_index.append(i)
    print("{} has {} lines".format(from_type, len(from_index)))
    print("{} has {} lines".format(to_type, len(to_index)))
    print("finding pair between {} and {}".format(from_type, to_type))
    best_pair = tfidf.make_pair(from_index, to_index)
    return best_pair


if __name__ == "__main__":
    dir_path = sys.argv[1].rstrip('/')
    tmpf = open(dir_path + '/all_data.text')
    print("data load ok")
    print("solving tf-idf:")
    tfidf = TFIDF(tmpf.readlines())
    tmpf.close()
    print("tf-idf ok")
    tmpf = open(dir_path + '/all_data.labels')
    labels = tmpf.readlines()
    print("labels load ok")
    print("totally, we have {} datas".format(len(labels)))
    
    from_type = sys.argv[2]
    to_type = sys.argv[3]
    best_pair = solve(from_type, to_type, tfidf, labels)

    with open(dir_path + "/pair.index_of_{}_{}".format(from_type, to_type), "w") as f:
        for p in best_pair:
            f.write("{},{},{}\n".format(p[0], p[1], p[2]))

    print("best pair match from {} to {}".format(from_type, to_type))



