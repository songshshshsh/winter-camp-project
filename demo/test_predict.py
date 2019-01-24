# coding: utf-8
from predict import getClass, getToken

def test():
	print('testing')
	#tmp = open('/home/bella/winter-camp-project/small_test.text').read().split('\n')[0]
	tmp = 'i love coding'
	print(tmp)
	class_name = getClass(tmp)
	print(class_name)
	transform = getToken(tmp, class_name, u'INTP')
	print(transform)

if __name__ == '__main__':
	test()