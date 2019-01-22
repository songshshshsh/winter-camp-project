import numpy as np
import pandas as pd
import pickle
import re
import codecs




# Load input dataset
mbti = pd.read_csv('data/mbti_1.csv')

# List of mbti types
type_labels = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP', 'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']

pattern_sep = r'\|\|\|'
subs_sep = r' '
# mbti['posts'] = mbti['posts'].replace(to_replace=pattern_sep, value=subs_sep, regex=True)
mbti['posts'] = mbti['posts'].str.lower()


def var_row(row):
    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)


mbti['words_per_comment'] = mbti['posts'].apply(lambda x: len(x.split())/50)
mbti['variance_of_word_counts'] = mbti['posts'].apply(lambda x: var_row(x))

focus = ['M', 'B', 'T', 'I']

for i in range(4):
    mbti[focus[i]] = mbti['type'].apply(lambda x: x[i])
    # print('focus on {}'.format(focus[i]))
    # print(mbti.groupby(focus[i]).agg({focus[i]:'count'}))


def is_delete(s):
    if "http://" in s:
        return True
    if "https://" in s:
        return True
    for x in s.split(' '):
        if x.isalpha():
            return False
    return True


# make data
dataset = []
for i in mbti['T'].unique():
    focus = mbti[mbti['T'] == i]
    content = []
    for sentence in focus['posts'].unique():
        content.extend(sentence.lstrip('\'').rstrip('\'').split("|||"))

    # f = codecs.open("data/sentence_of_{}.txt".format(i), 'a+', 'utf-8')
    for item in content:
        if not is_delete(item):
            # f.write(item + '\n')
            dataset.append([i, item])
    # f.close()
    print("focus on {}, and the size of dataset is {} now".format(i, len(dataset)))


from sklearn.feature_extraction.text import CountVectorizer

dataset = np.array(dataset)
#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
X = vectorizer.fit_transform(dataset[:,1])
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()

stoplist = [
    'very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
    'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
    'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
    'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
    'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
    'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
    'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
    'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
    'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
    'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
    'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
    'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which'
]


def is_bad_word(w):
    if not all(ord(c) < 128 for c in w):
        return True
    if w in stoplist:
        return True
    if w.isdigit():
        return True
    return False

remain_indices = []

for i in range(len(word)):
    if not is_bad_word(word[i]):
        remain_indices.append(i)

word_times = []
for i in range(X.shape[1]):
    if i % 100 == 0:
        print("{}/{}".format(i, X.shape[1]))
    word_times.append(np.sum(X[:, i]))

output = open('word_tf.pkl', 'wb')
# pickle.load(open('word_tf.pkl', 'rb'))
pickle.dump(word_times, output)
output.close()

use_word = []
indices = np.argsort(np.array(word_times) * -1)

for index in indices:
    if not is_bad_word(word[index]):
        use_word.append(word[index])
    if len(use_word) == 10000:
        break




