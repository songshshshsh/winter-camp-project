import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import stopwords

def hash_label(label):
	t = (label[0] == 'I') * 8 + (label[1] == 'N') * 4 + (label[2] == 'T') * 2 + (label[3] == 'J')
	return t

def process_posts(posts, words, stop):
	ban_lists = ['www', 'http', 'jpg', 'gif']
	replace_set = ['.', ',', '?', '!', '\\', '/', '*', '\'', '(', ')', ':', '[', ']', ';']
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
		current_post = []
		for word in post.lower().split():
			if word not in stop:
				current_post.append(word)
				if word not in words:
					words.add(word)
		if len(current_post) != 0:
			post_array.append(' '.join(current_post))
	return post_array

def save(filepath, label, posts):
	f_label = open(filepath + '.labels', 'w')
	f_text = open(filepath + '.text', 'w')
	for i in range(len(posts)):
		for text in posts[i]:
			f_label.write(str(label[i]) + '\n')
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
	words = set()
	train_label = np.array([hash_label(data) for data in raw_data['type'][:(int)(tot_data_size * train_ratio)]])
	train_posts = np.array([process_posts(data, words, stop) for data in raw_data['posts'][:(int)(tot_data_size * train_ratio)]])
	words_for_val_test = words.deepcopy()
	val_label = np.array([hash_label(data) for data in raw_data['type'][(int)(tot_data_size * train_ratio): (int)(tot_data_size * (train_ratio + val_ratio))]])
	val_posts = np.array([process_posts(data, words_for_val_test, stop) for data in raw_data['posts'][(int)(tot_data_size * train_ratio): (int)(tot_data_size * (train_ratio + val_ratio))]])
	test_label = np.array([hash_label(data) for data in raw_data['type'][(int)(tot_data_size * (train_ratio + val_ratio)):]])
	test_posts = np.array([process_posts(data, words_for_val_test, stop) for data in raw_data['posts'][(int)(tot_data_size * (train_ratio + val_ratio)):]])
	save('train', train_label, train_posts)
	save('val', val_label, val_posts)
	save('test', test_label, test_posts)
	f = open('vocab', 'w')
	for word in words:
		f.write(word + '\n')
	f.close()

def main():
	preprocess_data()

if __name__ == '__main__':
	main()