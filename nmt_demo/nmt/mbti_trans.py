import os
import sys


def mbti_trans(text, label_from, label_to):
    f = open("./tmp/content", "w")
    f.write(text.rstrip('\n')+'\n')
    f.close()
    run_code = 'python -m nmt.nmt --out_dir=./model_sets/{}2{}_model --inference_input_file=./tmp/content --inference_output_file=./tmp/result'.format(label_from, label_to)

    print(run_code)
    os.system(run_code)
    f = open("./tmp/result")
    ans = f.readlines()[0]
    f.close()

    return ans.rstrip('\n')

def getClass(text):
    return 'INFP'


if __name__ == "__main__":
    text = input("start: \n")
    f = input("from: \n")
    t = input("to: \n")
    ans = mbti_trans(text, f, t)
    print(ans)
