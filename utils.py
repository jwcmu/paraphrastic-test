import numpy as np
import torch
import random
from collections import Counter


def max_pool(x, lengths, gpu):
    mask = torch.arange(x.size(1), dtype=torch.int32)[None, :]
    v = torch.full(mask.size(), float("-inf"))
    z = torch.zeros(mask.size())
    if gpu:
        mask = mask.cuda()
        v = v.cuda()
        z = z.cuda()
    mask = mask < lengths[:, None]
    mask = torch.where((1 - mask.int()).bool(), v, z)
    return torch.max(x + mask.unsqueeze(2), dim=1)[0]

def mean_pool(x, lengths, gpu):
    mask = torch.arange(x.size(1), dtype=torch.int32)[None, :]
    if gpu:
        mask = mask.cuda()
    mask = mask < lengths[:, None]
    return (x * mask.unsqueeze(2)).sum(dim=1) / lengths[:, None]

def torchify_batch(batch, gpu):
    max_len = 0
    for i in batch:
        if len(i) > max_len:
            max_len = len(i)

    batch_len = len(batch)

    np_sents = np.zeros((batch_len, max_len), dtype='int32')
    np_lens = np.zeros((batch_len,), dtype='int32')

    for i, ex in enumerate(batch):
        np_sents[i, :len(ex)] = ex
        np_lens[i] = len(ex)

    idxs, lengths = torch.from_numpy(np_sents).type(torch.int32), \
                               torch.from_numpy(np_lens).type(torch.int32)

    if gpu:
        idxs = idxs.cuda()
        lengths = lengths.cuda()
    
    return idxs, lengths


class Batch(object):
    def __init__(self):
        self.g1 = None
        self.g1_l = None
        self.g2 = None
        self.g2_l = None
        self.p1 = None
        self.p1_l = None
        self.p2 = None
        self.p2_l = None
