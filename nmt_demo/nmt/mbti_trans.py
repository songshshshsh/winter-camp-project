import os
import sys


def mbti_trans(text, label_from, label_to):
    f = open("./tmp/content", "w")
    f.write(text.rstrip('\n')+'\n')
    f.close()
    os.system('. ./test.sh > log')
    f = open("./tmp/result")
    ans = f.readlines()[0]
    f.close()

    return ans.rstrip('\n')


if __name__ == "__main__":
    text = input("start: \n")
    ans = mbti_trans(text, 't', 'f')
    print(ans)
