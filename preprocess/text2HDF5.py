import sys
import numpy as np
import h5py
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data-file')
parser.add_argument('--sp-vocab-file')
parser.add_argument('--n-rows', type=int, default=2)

args = parser.parse_args()

f = open(args.data_file, 'r')
data_lines = f.readlines()

f = open(args.sp_vocab_file, 'r')
vocab_lines = f.readlines()

lis = []
vocab = OrderedDict()

for i in vocab_lines:
    i = i.strip()
    i = i.split('\t')
    vocab[i[0]] = len(vocab)

for i in data_lines:
    i = i.strip()
    i = i.split('\t')
    if len(i) != args.n_rows:
        continue
    arr = i[0].split()
    s1 = []
    for j in arr:
        s1.append(vocab[j])
    arr1 = np.array(s1, dtype="int32")
    arr = i[1].split()
    s2 = []
    for j in arr:
        s2.append(vocab[j])
    arr2 = np.array(s2, dtype="int32")
    lis.append((arr1, arr2))

arr = np.array(lis)
dt = h5py.vlen_dtype(np.dtype('int32'))

f = h5py.File(args.data_file.replace("txt","h5"), 'w')
f.create_dataset("data", data=arr, dtype=dt)

f = open(args.data_file.replace("txt","vocab"), 'w')
for i in vocab:
    f.write("{0}\t{1}\n".format(i,vocab[i]))
f.close()
