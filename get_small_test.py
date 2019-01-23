str = open('test.text').read().split('\n')[:3]
open('small_test.text', 'w').write('\n'.join(str) + '\n')
str = open('test.labels0').read().split('\n')[:3]
open('small_test.labels0', 'w').write('\n'.join(str) + '\n')
str = open('test.labels1').read().split('\n')[:3]
open('small_test.labels1', 'w').write('\n'.join(str) + '\n')
str = open('test.labels2').read().split('\n')[:3]
open('small_test.labels2', 'w').write('\n'.join(str) + '\n')
str = open('test.labels3').read().split('\n')[:3]
open('small_test.labels3', 'w').write('\n'.join(str) + '\n')

