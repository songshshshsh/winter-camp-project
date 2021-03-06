import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import stopwords

def hash_label(label):
	t = [(int)(label[0] == 'I'), (int)(label[1] == 'N'), (int)(label[2] == 'T'), (int)(label[3] == 'J')]
	return t

def process_posts(posts, words, stop):
	ban_lists = ['www', 'http', 'jpg', 'gif']
	symbol_set = ['.', ',', '?', '!', ':',]
	replace_set = ['\\', '/', '*', '(', ')', '[', ']', ';']
	post_array = []
	for post in posts.split('|||'):
		has_ban = False
		for ban in ban_lists:
			if ban in post.lower():
				has_ban = True
				break
		if has_ban:
			continue
		for replace_word in replace_set:
			post = post.replace(replace_word, '')
		for symbol in symbol_set:
			post = post.replace(symbol, ' ' + symbol + ' ')
		current_post = []
		for word in post.lower().split():
			if len(word) == 0:
				continue
			# if word not in stop:
			current_post.append(word)
			if word not in words:
				words[word] = 1
			else:
				words[word] += 1
		if len(current_post) != 0 and len(current_post) < 25:
			post_array.append(' '.join(current_post))
	return post_array

def save(filepath, label, posts, j):
	f_label = open(filepath + '.labels' + str(j), 'w')
	f_text = open(filepath + '.text', 'w')
	for i in range(len(posts)):
		print(len(posts[i]))
		for text in posts[i]:
			f_label.write(str(label[i][j]) + '\n')
			f_text.write(text + '\n')
	f_label.close()
	f_text.close()

def preprocess_data():
	raw_data = pd.read_csv('mbti_1.csv')
	raw_data = shuffle(raw_data)
	tot_data_size = len(raw_data)
	stop = stopwords.words('english')
	train_ratio = .7
	val_ratio = .1
	test_ratio = .2
	words = dict()
	train_label = np.array([hash_label(data) for data in raw_data['type'][:(int)(tot_data_size * train_ratio)]])
	train_posts = np.array([process_posts(data, words, stop) for data in raw_data['posts'][:(int)(tot_data_size * train_ratio)]])
	words_for_val_test = dict()
	val_label = np.array([hash_label(data) for data in raw_data['type'][(int)(tot_data_size * train_ratio): (int)(tot_data_size * (train_ratio + val_ratio))]])
	val_posts = np.array([process_posts(data, words_for_val_test, stop) for data in raw_data['posts'][(int)(tot_data_size * train_ratio): (int)(tot_data_size * (train_ratio + val_ratio))]])
	test_label = np.array([hash_label(data) for data in raw_data['type'][(int)(tot_data_size * (train_ratio + val_ratio)):]])
	test_posts = np.array([process_posts(data, words_for_val_test, stop) for data in raw_data['posts'][(int)(tot_data_size * (train_ratio + val_ratio)):]])
	for i in range(4):
		save('train', train_label, train_posts, i)
		save('val', val_label, val_posts, i)
		save('test', test_label, test_posts, i)
	f = open('vocab', 'w')
	for word in words.keys():
		if words[word] > 2:
			f.write(word + '\n')
	f.close()

def main():
	preprocess_data()

if __name__ == '__main__':
	main()
