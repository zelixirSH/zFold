import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from zfold.network.esm.multihead_attention import MultiheadAttention  # noqa
from zfold.network.esm.axial_attention import ColumnSelfAttention, RowSelfAttention

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)
except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)

def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x

class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, add_bias_kv=True, use_esm1b_layer_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim, self.attention_heads, add_bias_kv=add_bias_kv, add_zero_attn=False,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn

class AxialTransformerLayer(nn.Module):
    """ Implements an Axial MSA Transformer block.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2 ** 14,
        tied_attn = True,
        is_ncwh = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        self.tied_attn = tied_attn
        self.is_ncwh = is_ncwh

        if self.tied_attn:
            row_self_attention = RowSelfAttention(
                embedding_dim,
                num_attention_heads,
                dropout=dropout,
                max_tokens_per_msa=max_tokens_per_msa,
            )
        else:
            row_self_attention = ColumnSelfAttention(
                embedding_dim,
                num_attention_heads,
                dropout=dropout,
                max_tokens_per_msa=max_tokens_per_msa,
            )

        column_self_attention = ColumnSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
            activation_dropout=activation_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        self.row_self_attention    = self.build_residual(row_self_attention)
        self.column_self_attention = self.build_residual(column_self_attention)
        self.feed_forward_layer    = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module):
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
            self.dropout_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
        **kwargs,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """

        if self.is_ncwh:
            x = x.permute((2,3,0,1))

        if not self.tied_attn:
            # print('untied attn')
            _self_attn_mask = self_attn_mask.permute((1, 0, 2, 3)) if self_attn_mask is not None else None
            _self_attn_padding_mask = self_attn_padding_mask.permute((1, 0, 2, 3)) if self_attn_padding_mask is not None else None
            x, row_attn = self.row_self_attention(
                            x.permute((1,0,2,3)),
                            self_attn_mask=_self_attn_mask,
                            self_attn_padding_mask=_self_attn_padding_mask
            )
            x = x.permute((1,0,2,3))
        else:
            x, row_attn = self.row_self_attention(
                x,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )

        x, column_attn = self.column_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        x = self.feed_forward_layer(x)

        if self.is_ncwh:
            x = x.permute((2,3,0,1))

        if need_head_weights:
            return x, column_attn, row_attn
        else:
            return x

    def forward_two_crops(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        # self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask0: Optional[torch.Tensor] = None,
        self_attn_padding_mask1: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
        type='non_cross_attn',        # type = 'cross_attn'
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """

        if type == 'cross_attn':
            print(type)
            x_0, row_attn_0 = self.row_self_attention.forward_two_crops(x0, x1)
            x_1, row_attn_1 = self.row_self_attention.forward_two_crops(x1, x0)

        elif type == 'non_cross_attn':
            x_0, _ = self.row_self_attention(x0, self_attn_padding_mask=self_attn_padding_mask0)
            x_1, _ = self.row_self_attention(x1, self_attn_padding_mask=self_attn_padding_mask1)
            _, row_attn_0 = self.row_self_attention.forward_two_crops(x0, x1)
            _, row_attn_1 = self.row_self_attention.forward_two_crops(x1, x0)

        else:
            raise NotImplementedError

        row_attn = torch.cat([row_attn_0, row_attn_1.permute(0,1,3,2)], axis = 0)

        x_0, column_attn_0 = self.column_self_attention(x_0)# self_attn_padding_mask=self_attn_padding_mask0)
        x_1, column_attn_1 = self.column_self_attention(x_1)# self_attn_padding_mask=self_attn_padding_mask0)
        x_0 = self.feed_forward_layer(x_0)
        x_1 = self.feed_forward_layer(x_1)

        if need_head_weights:
            return x_0, x_1, row_attn
        else:
            return x_0, x_1

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, learned=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self, x):
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def make_positions(self, x):
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight, n_in = None, n_out = None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)

        if n_in is None and n_out is None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(torch.zeros(n_in, n_out))
            self.weight.data.copy_(weight.data)

        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class NormalizedResidualBlock(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        embedding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.layer = layer
        self.dropout_module = nn.Dropout(
            dropout,
        )
        self.layer_norm = ESM1bLayerNorm(self.embedding_dim)

    def forward(self, x, *args, **kwargs):
        residual = x

        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x

    #TODO
    # SOME BUGS
    def forward_two_crops_(self, x0, x1, *args, **kwargs):
        residual1 = x1
        x0 = self.layer_norm(x0)
        x1 = self.layer_norm(x1)
        outputs1 = self.layer.forward_two_crops(x0, x1, *args, **kwargs)

        if isinstance(outputs1, tuple):
            x1, *out1 = outputs1
        else:
            x1 = outputs1
            out1 = None

        x1 = self.dropout_module(x1)
        x1 = residual1 + x1

        if out1 is not None:
            return (x1,) + tuple(out1)
        else:
            return x1

    def forward_two_crops(self, x0, x1, *args, **kwargs):

        residual0 = x0

        x0 = self.layer_norm(x0)
        x1 = self.layer_norm(x1)

        outputs0 = self.layer.forward_two_crops(x0, x1, *args, **kwargs)

        if isinstance(outputs0, tuple):
            x0, *out0 = outputs0
        else:
            x0 = outputs0
            out0 = None

        x0 = self.dropout_module(x0)
        x0 = residual0 + x0

        if out0 is not None:
            return (x0,) + tuple(out0)
        else:
            return x0

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2 ** 14,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens_per_msa = max_tokens_per_msa
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout,
        )
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    import logging
    # x0 = torch.zeros([128, 32, 1, 768])  # w,h,n,c
    x0 = torch.zeros([3, 768, 128, 32])  # w,h,n,c
    net = RowSelfAttention(embed_dim=768, num_heads=12)
    print(net)

