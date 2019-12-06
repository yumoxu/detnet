# -*- coding: utf-8 -*-
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import copy


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
        Implement the PE function.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def clones(module, N):
    """
        Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, sublayer_type):
        """
            Apply residual connection to any sublayer with the same size.
            sublayer_type: 'attn' or 'ff'
        """
        assert sublayer_type in ('attn', 'ff')
        if sublayer_type == 'ff':
            return x + self.dropout(sublayer(self.norm(x)))

        res, attn = sublayer(self.norm(x))
        return x + self.dropout(res), attn


def attention(query, key, value, mask=None, dropout=None):
    """
        Scaled Dot Product Attention.
        q, k, v: d_batch * n_heads * n_words * d_k
        mask: d_batch * 1 * n_words * n_words or d_batch * 1 * n_words * 1 (1 is for broadcast)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # d_batch * n_heads * n_words * n_words
    # print('Shape: score - {0}, mask - {1}'.format(scores.size(), mask.size()))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # d_batch * n_heads * n_words * n_words
    if dropout is not None:
        p_attn = dropout(p_attn)  # d_batch * n_heads * n_words * n_words
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
            Take in model size and number of heads.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
            mask: 3d tensor, d_batch * n_words * (n_words or 1)
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # d_batch * 1 * n_words * (n_words or 1)
        d_batch = query.size(0)
        # print('Shape: query - {0}'.format(query.size()))

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # d_batch * n_words * n_heads * d_k
        query, key, value = [l(x).view(d_batch, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # print('Shape: query - {0}, key - {1}, value - {2}'.format(query.size(), key.size(), value.size()))

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # attn: d_batch * n_heads * n_words * n_words
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(d_batch, -1, self.h * self.d_k)
        # return self.linears[-1](x)
        return self.linears[-1](x), attn


class PositionwiseFeedForward(nn.Module):
    """
        Implements FFN equation.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderLayer(nn.Module):
    """
        Encoder is made up of self-attn and feed forward (defined below)
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
            mask: 3d tensor, d_batch * n_words * (n_words or 1)
        """
        x, attn = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask), sublayer_type='attn')
        return self.sublayer[1](x, self.feed_forward, sublayer_type='ff'), attn


class InsAttnLayer(nn.Module):
    """
        Encoder is made up of self-attn and feed forward (defined below)
    """
    def __init__(self, size, self_attn):
        super(InsAttnLayer, self).__init__()
        self.self_attn = self_attn
        self.size = size

    def forward(self, x, mask):
        """
            mask: 3d tensor, d_batch * n_words * (n_words or 1)
        """
        _, attn = self.self_attn(x, x, x, mask)
        return attn