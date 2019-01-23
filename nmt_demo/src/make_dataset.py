import numpy as np
import pandas as pd
import pickle
import copy
import math
import sys


def default_word(f):
    f.write('<unk>\n')
    f.write('<s>\n')
    f.write('</s>\n')

def write_file(word, f):
    f.write(word.rstrip('\n') + '\n')

if __name__ == "__main__":
    dir_path = sys.argv[1].rstrip('/')
    out_dir = sys.argv[2].rstrip('/')
    from_type = sys.argv[3]
    to_type = sys.argv[4]
    best_pair = open(dir_path + '/pair.index_of_{}_{}'.format(from_type, to_type)).readlines()
    content = open(dir_path + '/all_data.text').readlines()


    final_size = 0
    ft = open(out_dir + "/pair_data.from", "w")
    ff = open(out_dir + "/pair_data.to", "w")
    vt = open(out_dir + "/vo.from", "w")
    vf = open(out_dir + "/vo.to", "w")
    st = set()
    sf = set()
    
    for i in range(len(best_pair)):
        index = best_pair[i].split(',')
        if float(index[2]) < 0.6:
            continue
        ft.write(content[int(index[0])])
        ff.write(content[int(index[1])])

        for item in content[int(index[0])].split(' '):
            st.add(item.rstrip('\n'))

        for item in content[int(index[1])].split(' '):
            sf.add(item.rstrip('\n'))

    ft.close()
    ff.close()

    default_word(vt)
    default_word(vf)
    for item in st:
        write_file(item, vt)
    vt.close()

    for item in sf:
        write_file(item, vf)
    vf.close()

    print("dataset of {}2{} is made".format(from_type, to_type))

