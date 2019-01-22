import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def hash_label(label):
	# t = (label[0] == 'I') * 8 + (label[1] == 'N') * 4 + (label[2] == 'T') * 2 + (label[3] == 'J')
	t = label
	return t


def process_posts(posts, words):
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
	raw_data = pd.read_csv('data/mbti_1.csv')
	raw_data = shuffle(raw_data)
	tot_data_size = len(raw_data)
	words = set()
	label = np.array([data for data in raw_data['type']])
	posts = np.array([process_posts(data, words) for data in raw_data['posts']])
	save('data/data', label, posts)
	f = open('words', 'w')
	for word in words:
		f.write(word + '\n')
	f.close()


def main():
	preprocess_data()

if __name__ == '__main__':
	main()