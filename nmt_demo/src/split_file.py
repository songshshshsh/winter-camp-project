import numpy as np
import pandas as pd
import pickle
import copy
import math
import sys



if __name__ == "__main__":
    f = open(sys.argv[1])
    data = list(f.readlines())

    fout = open(sys.argv[2], "w")
    ratio_begin = float(sys.argv[3])
    ratio_end = float(sys.argv[4])

    data_size = float(len(data))
    

    for line in data[int(data_size * ratio_begin):int(data_size * ratio_end)]:
        fout.write(line)
    f.close()
    fout.close()

