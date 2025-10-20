import numpy as np
import torch
import ml_collections
from box import Box

from zfold.config import XFOLD_CONFIG, update_config
from zfold.zfoldnet import *

def unitest_zfold_define():
    #hard code for unitest
    XFOLD_CONFIG.basic.d_msa = 256
    XFOLD_CONFIG.basic.d_pair = 192
    XFOLD_CONFIG.basic.n_head_msa = 256 // 32
    XFOLD_CONFIG.basic.n_head_pair = 192 // 32

    XFOLD_CONFIG.fas_bert.enable = False
    XFOLD_CONFIG.modules.extra_msa_stack.enable = False
    XFOLD_CONFIG.modules.extra_msa_stack.d_msa = 49
    XFOLD_CONFIG.modules.extra_msa_stack.dim = 64

    XFOLD_CONFIG.templ.af2 = True
    XFOLD_CONFIG.templ.d_t1d = 57
    XFOLD_CONFIG.templ.d_t2d = 88

    is_cuda = True
    net = XFold2D(config = XFOLD_CONFIG)
    net.eval()
    print(net)

    B, K, L,  = 1, 93, 78 #batch_size, num_alignments, seq_len
    T, d_t1d, d_t2d = 4, XFOLD_CONFIG.templ.d_t1d, XFOLD_CONFIG.templ.d_t2d
    t1ds = torch.FloatTensor(np.ones((B, T, L, d_t1d)))    #.cuda() # n,d,c,c
    t2ds = torch.FloatTensor(np.ones((B, T, L, L, d_t2d))) #.cuda() # n,d,c,c
    tokens = torch.LongTensor(np.ones((B, K, L)))          #.cuda() # n,d,c,c
    print(t1ds.shape, t2ds.shape, tokens.shape)

    if is_cuda:
        net = net.cuda()
        t1ds = t1ds.cuda()
        t2ds = t2ds.cuda()
        tokens = tokens.cuda()

    out = net.forward(tokens = tokens,
                      t1ds = t1ds,
                      t2ds = t2ds)

    for key in out.keys():
        print(out[key].shape)

if __name__ == '__main__':
    unitest_zfold_define()
