import torch.nn as nn
from zfold.utils import exists
from zfold.network.modules import DirectMultiheadAttention, FeedForwardLayer
from zfold.network.attention import Attention, TriangleMultiplicativeModule
from zfold.network.af2_smod.net.inv_pnt_attn import InvPntAttn

# Initialization
def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

def init_zero_mlp(module):
    if isinstance(module, FeedForwardLayer):
        init_zero_(module.linear2)
    elif isinstance(module, Attention):
        init_zero_(module.to_out)
    elif isinstance(module, TriangleMultiplicativeModule):
        init_zero_(module.to_out)
    elif isinstance(module, DirectMultiheadAttention):
        init_zero_(module.proj_out)
    elif isinstance(module, InvPntAttn):
        init_zero_(module.linear_s)
