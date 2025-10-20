import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from zfold.network.rpe import RotaryEmbedding, apply_rotary_pos_emb, RPE_T5
from zfold.utils import exists, default
# token shifting
from zfold.network.activation import get_activation_fn

class DConv_project(nn.Module):
    def __init__(self, d_model_in, d_model_out):
        super(DConv_project, self).__init__()
        self.linear = nn.Linear(d_model_in, d_model_out, bias = False)
        self.d_conv = nn.Conv1d(d_model_out, d_model_out, kernel_size = 3, padding = 1, groups = d_model_out)

    def forward(self, x):
        x = self.linear(x)

        x = x.permute(0, 2, 1).contiguous()  # prep for convolution layer
        x = self.d_conv(x)                   # (B, C, L)
        x = x.permute(0, 2, 1).contiguous()  # (B, L, C)

        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, pb_relax = False):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        self.pb_replax = pb_relax

    def forward(self, x):

        if self.pb_replax:
            x = x / (x.abs().max().detach() / 8)

        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2*(x-mean)
        x /= std
        x += self.b_2
        return x

class FeedForwardLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 p_drop = 0.1,
                 d_model_out = None,
                 activation = "relu",
                 is_post_act_ln = False,
                 pb_relax = False,
                 **unused,
                 ):

        super(FeedForwardLayer, self).__init__()
        d_model_out = default(d_model_out, d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.post_act_ln = LayerNorm(d_ff, pb_relax=pb_relax) if is_post_act_ln else nn.Identity()
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_model_out)
        self.activation = get_activation_fn(activation=activation)

    def forward(self, src):
        src = self.linear2(self.dropout(self.post_act_ln(self.activation(self.linear1(src)))))
        return src

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        n_dims,
        n_heads = 8,
        n_dim_heads = 64,
        is_gating = True,
        is_head_scale = False,
        p_attn_drop=0.,
        rpe_type = None, # "RPE_T5", "Rotary"
        is_stable_softmax=True,
        proj_type = 'linear',#'DConv_project'
    ):
        super().__init__()

        inner_dim = n_dim_heads * n_heads
        self.heads = n_heads
        self.scale = n_dim_heads ** -0.5

        self.is_gating = is_gating
        self.is_stable_softmax = is_stable_softmax

        self.rpe_type = rpe_type

        if self.rpe_type == 'RPE_T5':
            self.rpe = RPE_T5(num_buckets=32, n_heads=n_heads, max_distance=128)
        elif self.rpe_type == 'Rotary':
            rotary_emb_dim = max(n_dim_heads // 2, 32)
            self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
        elif self.rpe_type is not None:
            raise NotImplementedError

        # head scaling
        self.is_head_scale = is_head_scale
        if self.is_head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, n_heads, 1, 1))

        if self.is_gating:
            self.gating = nn.Linear(n_dims, inner_dim)
            nn.init.constant_(self.gating.weight, 0.)
            nn.init.constant_(self.gating.bias, 1.)

        self.proj_type = proj_type
        if self.proj_type == 'DConv_project':
            self.to_q = DConv_project(n_dims, inner_dim)
            self.to_k = DConv_project(n_dims, inner_dim)
            self.to_v = DConv_project(n_dims, inner_dim)
        elif self.proj_type == 'linear':
            self.to_q = nn.Linear(n_dims, inner_dim, bias = False)
            self.to_kv = nn.Linear(n_dims, inner_dim * 2, bias = False)
        else:
            raise NotImplementedError

        self.to_out = nn.Linear(inner_dim, n_dims)
        self.attn_drop = nn.Dropout(p_attn_drop)

    def forward(self, x, mask = None, attn_bias = None, context = None, context_mask = None, tie_dim = None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)

        if self.proj_type == 'DConv_project':
            q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        else:
            q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if self.rpe_type == 'Rotary':
            max_rotary_emb_length = x.shape[1]
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
            ql, kl, vl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl))
            q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))

        # scale
        q = q * self.scale

        # query / key similarities
        if exists(tie_dim):
            # as in the paper, for the extra MSAs, they average the queries along the rows of the MSAs
            # they named this particular module MSAColumnGlobalAttention
            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r = tie_dim), (q, k))
            q = q.mean(dim = 1)

            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)
        if exists(attn_bias):
            dots = dots + attn_bias

        # add T5 relative pos embedding
        if self.rpe_type == 'RPE_T5':
            rpe_bias = self.rpe.compute_bias(query_length=i, key_length=j)
            dots = dots + rpe_bias

        # masking
        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2], device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention
        if self.is_stable_softmax:
            # use stable softmax in attention
            dots = dots - dots.max(dim=-1, keepdims=True).values

        attn = dots.softmax(dim = -1)
        attn = self.attn_drop(attn)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        if self.is_head_scale:
            out = out * self.head_scale_params

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating
        if self.is_gating:
            gates = self.gating(x)
            out = out * gates.sigmoid()

        # combine to out
        out = self.to_out(out)

        return out

class AxialAttention(nn.Module):
    def __init__(
        self,
        n_dims,
        n_heads,
        is_row_attn = True,
        is_col_attn = True,
        n_edge_dims = None,
        is_accept_edges = False,
        is_edge_norm = True,
        is_global_query_attn = False,
        is_use_ln = True,
        is_sandwich_norm = False,
        n_layer_shift_tokens = 0,
        pb_relax = False,
        **kwargs
    ):
        super().__init__()
        assert not (not is_row_attn and not is_col_attn), 'row or column attention must be turned on'

        self.row_attn = is_row_attn
        self.col_attn = is_col_attn
        self.global_query_attn = is_global_query_attn

        self.norm = LayerNorm(n_dims, pb_relax = pb_relax) if is_use_ln else nn.Identity()
        self.norm_edge = LayerNorm(n_edge_dims, pb_relax = pb_relax) \
            if is_accept_edges and (is_edge_norm and is_use_ln) else nn.Identity()
        self.postnorm = LayerNorm(n_dims, pb_relax = pb_relax) if is_sandwich_norm else nn.Identity()

        self.attn = Attention(n_dims = n_dims, n_heads = n_heads, **kwargs)

        assert not (is_col_attn and (n_layer_shift_tokens > 0)), print(is_col_attn, n_layer_shift_tokens)

        if n_layer_shift_tokens > 0:
            shift_range_upper = n_layer_shift_tokens + 1
            shift_range_lower = - n_layer_shift_tokens
            self.attn = ShiftTokens(range(shift_range_lower, shift_range_upper), self.attn)

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(n_edge_dims, n_heads, bias = False),
            Rearrange('b i j h -> b h i j')
        ) if is_accept_edges else None

    def forward(self, x, edges = None, mask = None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w, d = x.shape

        x = self.norm(x)

        # axial attention
        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'

        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'

        x = rearrange(x, input_fold_eq)

        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(self.norm_edge(edges))
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = axial_dim)

        tie_dim = axial_dim if self.global_query_attn else None

        out = self.attn(x, mask = mask, attn_bias = attn_bias, tie_dim = tie_dim)
        out = rearrange(out, output_fold_eq, h = h, w = w)

        return self.postnorm(out)

class MsaAttentionBlock(nn.Module):
    def __init__(
        self,
        msa_embed_dim,
        heads,
        dim_head,
        r_ff = 4,
        pair_embed_dim = None,
        p_drop = 0.,
        p_attn_drop = 0.,
        p_layer_drop = 0.,
        is_rezero = False,
        is_use_ln = True,
        rpe_type = None,
        is_sandwich_norm = False,
        is_head_scale = False,
        is_post_act_ln = False,
        n_layer_shift_tokens = 0,
        activation = 'relu',
        proj_type = 'linear',
        pb_relax = False,
        is_global_query_attn = False,
        **kwargs
    ):
        super().__init__()

        # Define the Resisdual Weight for ReZero
        self.resweight = torch.nn.Parameter(torch.Tensor([0]), requires_grad=True) if is_rezero else 1

        self.layer_drop = DropPath(p_layer_drop) if p_layer_drop > 0. else nn.Identity()

        self.row_attn = AxialAttention(n_dims = msa_embed_dim,
                                       n_heads = heads,
                                       n_dim_heads = dim_head,
                                       is_row_attn = True,
                                       is_global_query_attn = is_global_query_attn,
                                       is_col_attn = False,
                                       n_edge_dims = pair_embed_dim,
                                       is_accept_edges = exists(pair_embed_dim),
                                       is_edge_norm = exists(pair_embed_dim),
                                       p_attn_drop = p_attn_drop,
                                       is_use_ln = is_use_ln,
                                       rpe_type = rpe_type,
                                       n_layer_shift_tokens = n_layer_shift_tokens,
                                       is_sandwich_norm = is_sandwich_norm,
                                       is_head_scale = is_head_scale,
                                       proj_type = proj_type,
                                       pb_relax = pb_relax,
                                       )

        self.col_attn = AxialAttention(n_dims = msa_embed_dim,
                                       n_heads = heads,
                                       n_dim_heads = dim_head,
                                       is_row_attn = False,
                                       is_col_attn = True,
                                       n_edge_dims = pair_embed_dim,
                                       is_accept_edges = False,
                                       is_edge_norm = False,
                                       p_attn_drop = p_attn_drop,
                                       is_use_ln = is_use_ln,
                                       rpe_type = rpe_type,
                                       n_layer_shift_tokens = 0,
                                       is_sandwich_norm = is_sandwich_norm,
                                       is_head_scale = is_head_scale,
                                       proj_type = proj_type,
                                       pb_relax = pb_relax,
                                       )

        self.ffn_ln = LayerNorm(msa_embed_dim, pb_relax = pb_relax) if is_use_ln else nn.Identity()
        self.ffn_postln = LayerNorm(msa_embed_dim, pb_relax = pb_relax) if is_sandwich_norm else nn.Identity()
        self.ffn = FeedForwardLayer(msa_embed_dim,
                                    msa_embed_dim * r_ff,
                                    p_drop = p_drop,
                                    is_post_act_ln = is_post_act_ln,
                                    activation = activation,
                                    )

    def forward(
        self,
        x,
        mask = None,
        pair = None
    ):
        x = self.resweight * self.layer_drop(self.row_attn(x, mask = mask, edges = pair)) + x
        x = self.resweight * self.layer_drop(self.col_attn(x, mask = mask)) + x
        x = self.resweight * self.layer_drop(self.ffn_postln(self.ffn(self.ffn_ln(x)))) + x

        return x

class PairwiseAttentionBlock(nn.Module):
    def __init__(
        self,
        pair_embed_dim,
        heads,
        dim_head,
        msa_embed_dim = None,
        is_edge_norm = True,
        is_sandwich_norm = False,
        p_attn_drop = 0.,
        p_drop = 0.,
        p_layer_drop = 0.,
        r_ff = 4,
        is_rezero = False,
        is_use_ln = True,
        is_post_act_ln = False,
        n_layer_shift_tokens = 0,
        activation = 'relu',
        proj_type='linear',
        pb_relax = False,
        **unused,
    ):
        super().__init__()

        self.msa_embed_dim = msa_embed_dim

        assert n_layer_shift_tokens == 0, print('No token shift in Pair feature update')
        # Define the Resisdual Weight for ReZero
        self.resweight = torch.nn.Parameter(torch.Tensor([0]), requires_grad=True) if is_rezero else 1

        self.layer_drop = DropPath(p_layer_drop) if p_layer_drop > 0. else nn.Identity()

        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = pair_embed_dim,
                                                                       is_use_ln = is_use_ln,
                                                                       mix = 'outgoing',
                                                                       is_sandwich_norm = is_sandwich_norm,
                                                                       )
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = pair_embed_dim,
                                                                      is_use_ln = is_use_ln,
                                                                      mix = 'ingoing',
                                                                      is_sandwich_norm = is_sandwich_norm,
                                                                      )

        self.triangle_attention_outgoing = AxialAttention(n_dims = pair_embed_dim,
                                                          n_heads = heads,
                                                          n_dim_heads = dim_head,
                                                          is_row_attn = True,
                                                          is_col_attn = False,
                                                          is_accept_edges = True,
                                                          n_edge_dims = pair_embed_dim,
                                                          is_edge_norm = is_edge_norm,
                                                          p_attn_drop = p_attn_drop,
                                                          is_use_ln = is_use_ln,
                                                          is_sandwich_norm = is_sandwich_norm,
                                                          proj_type = proj_type,
                                                          pb_relax=pb_relax,
                                                          )
        self.triangle_attention_ingoing = AxialAttention(n_dims = pair_embed_dim,
                                                         n_heads = heads,
                                                         n_dim_heads = dim_head,
                                                         is_row_attn = False,
                                                         is_col_attn = True,
                                                         is_accept_edges = True,
                                                         n_edge_dims = pair_embed_dim,
                                                         is_edge_norm = is_edge_norm,
                                                         p_attn_drop = p_attn_drop,
                                                         is_use_ln = is_use_ln,
                                                         is_sandwich_norm = is_sandwich_norm,
                                                         proj_type = proj_type,
                                                         pb_relax = pb_relax,
                                                         )

        self.ffn_ln = LayerNorm(pair_embed_dim, pb_relax = pb_relax) if is_use_ln else nn.Identity()
        self.ffn_postln = LayerNorm(pair_embed_dim, pb_relax = pb_relax) if is_sandwich_norm else nn.Identity()
        self.ffn = FeedForwardLayer(pair_embed_dim,
                                    pair_embed_dim*r_ff,
                                    p_drop = p_drop,
                                    is_post_act_ln = is_post_act_ln,
                                    activation=activation,
                                    )

    def forward(
        self,
        x,
        mask = None,
        msa_repr = None,
    ):

        x = self.resweight * self.layer_drop(self.triangle_multiply_outgoing(x, mask = mask)) + x
        x = self.resweight * self.layer_drop(self.triangle_multiply_ingoing(x, mask = mask)) + x
        x = self.resweight * self.layer_drop(self.triangle_attention_outgoing(x, edges = x, mask = mask)) + x
        x = self.resweight * self.layer_drop(self.triangle_attention_ingoing(x, edges = x, mask = mask)) + x

        x = self.resweight * self.layer_drop(self.ffn_postln(self.ffn(self.ffn_ln(x)))) + x

        return x

class DirectMultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, heads, p_attn_drop = 0.1):
        super(DirectMultiheadAttention, self).__init__()
        self.heads = heads
        self.proj_pair = nn.Linear(d_in, heads)
        self.proj_msa = nn.Linear(d_out, d_out)
        self.proj_out = nn.Linear(d_out, d_out)
        self.attn_drop = nn.Dropout(p_attn_drop)

    def forward(self, src, tgt):
        B, N, L = tgt.shape[:3]
        attn_map = F.softmax(self.proj_pair(src), dim=1).permute(0,3,1,2) # (B, h, L, L)
        attn_map = self.attn_drop(attn_map).unsqueeze(1)

        # apply attention
        value = self.proj_msa(tgt).permute(0,3,1,2).contiguous().view(B, -1, self.heads, N, L) # (B,-1, h, N, L)
        tgt = torch.matmul(value, attn_map).view(B, -1, N, L).permute(0,2,3,1) # (B,N,L,K)
        tgt = self.proj_out(tgt)
        return tgt

class DirectEncoderLayer(nn.Module):
    def __init__(self,
                 heads,
                 d_in,
                 d_out,
                 d_ff,
                 is_rezero = False,
                 p_drop = 0.1,
                 p_layer_drop = 0.,
                 p_attn_drop = 0.0,
                 is_use_ln = True,
                 is_sandwich_norm = False,
                 activation = 'relu',
                 pb_relax = False,
                 ):

        super(DirectEncoderLayer, self).__init__()
        self.resweight = torch.nn.Parameter(torch.Tensor([0]), requires_grad=True) if is_rezero else 1
        #
        self.layer_drop = DropPath(p_layer_drop) if p_layer_drop > 0. else nn.Identity()
        # multihead attention
        self.attn = DirectMultiheadAttention(d_in, d_out, heads, p_attn_drop=p_attn_drop)
        # feedforward
        self.ff = FeedForwardLayer(d_out,
                                   d_ff,
                                   p_drop=p_drop,
                                   is_post_act_ln = is_sandwich_norm,
                                   activation = activation,
                                   )

        # LayerNorm
        self.norm = LayerNorm(d_in, pb_relax = pb_relax) if is_use_ln else nn.Identity()
        self.norm1 = LayerNorm(d_out, pb_relax = pb_relax) if is_use_ln else nn.Identity()
        self.norm2 = LayerNorm(d_out, pb_relax = pb_relax) if is_use_ln else nn.Identity()
        self.post_norm = LayerNorm(d_out, pb_relax = pb_relax) if is_sandwich_norm else nn.Identity()

    def forward(self, src, tgt):
        # Input:
        #  For pair to msa: src=pair (B, L, L, C), tgt=msa (B, N, L, K)
        B, N, L = tgt.shape[:3]

        # get attention map
        if True:
            src = 0.5 * (src + src.permute(0,2,1,3))

        src = self.norm(src)
        tgt2 = self.norm1(tgt)
        tgt2 = self.attn(src, tgt2)
        tgt = tgt + self.resweight * self.layer_drop(tgt2)

        # feed-forward
        tgt2 = self.norm2(tgt.view(B*N,L,-1)).view(B,N,L,-1)
        tgt2 = self.post_norm(self.ff(tgt2))
        tgt = tgt + self.resweight * self.layer_drop(tgt2)

        return tgt

class TriangleMultiplicativeModuleNew(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        is_use_ln = True,
        is_sandwich_norm = False,
        mix = 'ingoing',
        pb_relax = False,
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = LayerNorm(dim, pb_relax=pb_relax) if is_use_ln else nn.Identity()
        self.postnorm = LayerNorm(dim, pb_relax=pb_relax) if is_sandwich_norm else nn.Identity()

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = LayerNorm(hidden_dim, pb_relax=pb_relax) if is_use_ln else nn.Identity()
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'

        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        '''
        # TODO avoid inf
        if not self.training:
            left /= x.shape[1]
        out = einsum(self.mix_einsum_eq, left, right)
        '''

        # TODO
        dtype = right.dtype
        out = einsum(self.mix_einsum_eq, left.float(), right.float()) / x.shape[1]
        out = out.to(dtype=dtype)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.postnorm(self.to_out(out))

class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        is_use_ln = True,
        is_sandwich_norm = False,
        mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim) if is_use_ln else nn.Identity()
        self.postnorm = nn.LayerNorm(dim) if is_sandwich_norm else nn.Identity()

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim) if is_use_ln else nn.Identity()
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'

        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        '''
        # TODO avoid inf
        if not self.training:
            left /= x.shape[1]
        out = einsum(self.mix_einsum_eq, left, right)
        '''

        # TODO
        dtype = right.dtype
        out = einsum(self.mix_einsum_eq, left.float(), right.float()) / x.shape[1]
        out = out.to(dtype=dtype)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.postnorm(self.to_out(out))

if __name__ == '__main__':
    import numpy as np
    pair_fea = torch.zeros([4, 48, 48, 288])  # n,d,c,c
    msa_fea  = torch.zeros([4, 32, 48, 384])  # n,r,c,d

    param = {
            'pair_embed_dim': 288,
            'heads':          8,
            'dim_head':       64,
            'p_dropout':      0.0,
            'p_layer_drop':   0.1,
            'activation':     "relu",
            'is_gating':      False,
            'is_rezero':      False,
            'rpe_type':       "Rotary",
            'is_head_scale':  True,
            'proj_type':      'DConv_project',
            'is_sandwich_norm': True,
            }
    net = PairwiseAttentionBlock(**param)
    print(net)

    pair_fea = net(pair_fea, msa_repr = msa_fea)
    print('pair', pair_fea.shape)

    param = {
            'msa_embed_dim': 384,
            'heads':         12,
            'dim_head':      64,
            'p_dropout':     0.0,
            'p_layer_drop':  0.1,
            'activation':    "relu",
            'is_gating':     False,
            'is_rezero':     False,
            'rpe_type':      "Rotary",
            'n_layer_shift_tokens': 1,
            'is_head_scale':    True,
            }
    net = MsaAttentionBlock(**param)
    print(net)
    msa_fea = net(msa_fea)
    print('pair', msa_fea.shape)

    net = TriangleMultiplicativeModule(dim = 384, hidden_dim = 128, mix = 'ingoing')
    print(net)
