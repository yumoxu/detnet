# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from frame import transformer as trans


class Encoder(nn.Module):
    """
        Core encoder is a stack of N layers
    """
    def __init__(self, layer, n_layers, ins_attn, ins_attn_layer=None, d_embed=None):
        super(Encoder, self).__init__()
        self.layers = trans.clones(layer, n_layers)
        self.norm = trans.LayerNorm(layer.size)
        self.ins_attn = ins_attn
        self.d_embed = d_embed
        if ins_attn == 'key':
            self.attn_key = nn.Linear(d_embed, 1, bias=False)

        self.ins_attn_layer = ins_attn_layer

    def build_ins_attn(self, embed_tensor, mask_mat):
        softmax = nn.Softmax(dim=-1)

        if self.ins_attn == 'key':
            attn_scores = self.attn_key(embed_tensor).squeeze(-1)  # d_batch * n_words
        elif self.ins_attn == 'multi':  # create a new multi-attn instead of using the attn at the last layer
            attn = self.ins_attn_layer(embed_tensor, mask_mat)  # n_sents * n_heads * n_words * n_words
            n_head, n_word = attn.size()[1:3]
            attn_scores = torch.sum(torch.sum(attn, dim=1), dim=1) / math.sqrt(n_head * n_word)  # d_batch * n_words
        else:  # single
            scores = torch.matmul(embed_tensor, embed_tensor.transpose(-2, -1)) / math.sqrt(self.d_embed)  # n_sents * n_words * n_words
            scores = scores.masked_fill(mask_mat == 0, -1e9)  # n_sents * n_words * n_words
            n_word = mask_mat.size(1)
            attn_scores = torch.sum(scores, dim=1) / math.sqrt(n_word)

        attn_scores = attn_scores.masked_fill(mask_mat.squeeze(-1) == 0, -1e9)
        attn_mat = softmax(attn_scores)

        return attn_mat

    def encode_3d_embeds(self, embed_tensor, mask_mat):
        """
            embed_tensor could be
                word_embed_tensor: n_sents * n_words * d_embed
                OR
                batched_sent_embed_tensor, d_batch * n_sents * d_embed

            mask_mat, 2d mat, could be
                word_mask_mat: n_sents * n_words
                OR
                batched_sent_mask_mat: d_batch * n_sents
        """
        n_ele_in_batch = torch.sum(mask_mat, dim=-1, keepdim=True)  # d_batch, 1. double.
        n_ele_in_batch[n_ele_in_batch == float(0)] = float(1)  # for div, replacing 0 with 1 does not matter

        mask_mat = mask_mat.unsqueeze(-1)  # 3d mat, n_sents * n_words * 1 or n_batch * n_sents * 1

        for layer in self.layers:
            # print('Shape: embed_tensor: {}'.format(embed_tensor.size()))
            embed_tensor, _ = layer(embed_tensor, mask_mat)  # just use embed_tensor and attn at the last layer
            # embed_tensor, attn = layer(embed_tensor, mask_mat)  # just use embed_tensor and attn at the last layer

        if self.norm:
            embed_tensor = self.norm(embed_tensor)

        higher_level_embed_mat = torch.sum(embed_tensor, dim=-2)  # d_batch * d_embed
        denom = torch.sqrt(n_ele_in_batch).expand_as(higher_level_embed_mat).float()  # d_batch, 1 => d_batch, d_embed

        higher_level_embed_mat = higher_level_embed_mat / denom

        # print('higher_level_embed_mat: {}'.format(higher_level_embed_mat))
        enc_res = {
            'embed_tensor': embed_tensor,
            'higher_level_embed_mat': higher_level_embed_mat,
        }

        if self.ins_attn:  # discard some models e.g. HierNet
            attn_mat = self.build_ins_attn(embed_tensor, mask_mat)
            enc_res['attn_mat'] = attn_mat

        return enc_res

    def forward(self, *input):
        raise NotImplementedError


class WordEncoder(Encoder):
    """
        Core encoder is a stack of N layers
    """
    def __init__(self, layer, n_layers, ins_attn, ins_attn_layer, d_embed):
        super(WordEncoder, self).__init__(layer, n_layers, ins_attn, ins_attn_layer, d_embed)

    def forward(self, word_embeds, word_mask):
        """
            word_embeds: d_batch * n_sents * n_words * d_embed or 1 * n_doms * n_sents * n_words * d_embed
            word_mask: d_batch * n_sents * n_words or 1 * n_doms * n_sents * n_words
        """
        word_reps_list, sent_embeds_list = list(), list()
        if self.ins_attn:
            word_attn_list = list()

        for word_embed_tensor, mask_mat in zip(word_embeds, word_mask):
            # treat n_sent as d_batch in the loop
            enc_res = self.encode_3d_embeds(word_embed_tensor, mask_mat)

            word_reps_list.append(enc_res['embed_tensor'])
            sent_embeds_list.append(enc_res['higher_level_embed_mat'])
            if self.ins_attn:
                word_attn_list.append(enc_res['attn_mat'])

        word_reps = torch.stack(word_reps_list)
        sent_embeds = torch.stack(sent_embeds_list)

        word_enc_res = {
            'word_reps': word_reps,
            'sent_embeds': sent_embeds,
        }

        if self.ins_attn:
            word_attn = torch.stack(word_attn_list)
            word_enc_res['word_attn'] = word_attn  # d_batch * n_sents * n_words

        return word_enc_res


class SentEncoder(Encoder):
    def __init__(self, layer, n_layers, ins_attn, ins_attn_layer, d_embed, pos_enc=None):
        super(SentEncoder, self).__init__(layer, n_layers, ins_attn, ins_attn_layer, d_embed)
        self.pos_enc = pos_enc

    def forward(self, sent_embeds, sent_mask):
        """
            sent_embeds: n_batch * n_sents * d_embed
            sent_mask: n_batch * n_sents
        """
        if self.pos_enc:
            sent_embeds = self.pos_enc(sent_embeds)

        enc_res = self.encode_3d_embeds(sent_embeds, sent_mask)

        sent_enc_res = {
            'sent_reps': enc_res['embed_tensor'],
        }

        if self.ins_attn:
            sent_enc_res['sent_attn'] = enc_res['attn_mat']  # d_batch * n_sents

        return sent_enc_res


class WordEncoderWithAttnRep(Encoder):
    """
        Core encoder is a stack of N layers.

        Features:
            [1] Use word reps * attn to generate sent embeds.
    """
    def __init__(self, layer, n_layers, ins_attn, ins_attn_layer, d_embed):
        super(WordEncoderWithAttnRep, self).__init__(layer, n_layers, ins_attn, ins_attn_layer, d_embed)

    def encode_3d_embeds(self, embed_tensor, mask_mat):
        """
            embed_tensor could be
                word_embed_tensor: n_sents * n_words * d_embed
                OR
                batched_sent_embed_tensor, d_batch * n_sents * d_embed

            mask_mat, 2d mat, could be
                word_mask_mat: n_sents * n_words
                OR
                batched_sent_mask_mat: d_batch * n_sents
        """
        n_ele_in_batch = torch.sum(mask_mat, dim=-1, keepdim=True)  # d_batch, 1. double.
        n_ele_in_batch[n_ele_in_batch == float(0)] = float(1)  # for div, replacing 0 with 1 does not matter

        higher_level_embed_mat = torch.sum(embed_tensor, dim=-2)  # d_batch * d_embed
        denom = torch.sqrt(n_ele_in_batch).expand_as(higher_level_embed_mat).float()  # d_batch, 1 => d_batch, d_embed
        higher_level_embed_mat = higher_level_embed_mat / denom

        mask_mat = mask_mat.unsqueeze(-1)  # 3d mat, n_sents * n_words * 1 or n_batch * n_sents * 1

        for layer in self.layers:
            # print('Shape: embed_tensor: {}'.format(embed_tensor.size()))
            embed_tensor, _ = layer(embed_tensor, mask_mat)  # just use embed_tensor and attn at the last layer
            # embed_tensor, attn = layer(embed_tensor, mask_mat)  # just use embed_tensor and attn at the last layer

        if self.norm:
            embed_tensor = self.norm(embed_tensor)

        # print('higher_level_embed_mat: {}'.format(higher_level_embed_mat))
        enc_res = {
            'embed_tensor': embed_tensor,
            'higher_level_embed_mat': higher_level_embed_mat,
        }

        if self.ins_attn:  # discard some models e.g. HierNet
            attn_mat = self.build_ins_attn(embed_tensor, mask_mat)  # n_sents * n_words
            enc_res['attn_mat'] = attn_mat

            embed_tensor = embed_tensor.transpose(1, 2).contiguous()  # n_sents * d_embed * n_words
            attn_mat = attn_mat.unsqueeze(-1)  # n_sents * n_words * 1
            higher_level_embed_mat = torch.matmul(embed_tensor, attn_mat).squeeze(-1)  # n_sents * n_dom
            enc_res['higher_level_embed_mat'] = higher_level_embed_mat

        return enc_res

    def forward(self, word_embeds, word_mask):
        """
            word_embeds: d_batch * n_sents * n_words * d_embed or 1 * n_doms * n_sents * n_words * d_embed
            word_mask: d_batch * n_sents * n_words or 1 * n_doms * n_sents * n_words
        """
        word_reps_list = list()
        if self.ins_attn:
            word_attn_list = list()
            sent_embeds_list = list()

        for word_embed_tensor, mask_mat in zip(word_embeds, word_mask):
            # treat n_sent as d_batch in the loop
            enc_res = self.encode_3d_embeds(word_embed_tensor, mask_mat)

            word_reps_list.append(enc_res['embed_tensor'])
            # sent_embeds_list.append(enc_res['higher_level_embed_mat'])
            if self.ins_attn:
                word_attn_list.append(enc_res['attn_mat'])
                sent_embeds_list.append(enc_res['higher_level_embed_mat'])

        word_reps = torch.stack(word_reps_list)
        # sent_embeds = torch.stack(sent_embeds_list)

        word_enc_res = {
            'word_reps': word_reps,
            # 'sent_embeds': sent_embeds,
        }

        if self.ins_attn:
            word_attn = torch.stack(word_attn_list)
            word_enc_res['word_attn'] = word_attn  # d_batch * n_sents * n_words
            word_enc_res['sent_embeds'] = torch.stack(sent_embeds_list)  # d_batch * n_sents * d_embed

        return word_enc_res


class StefanosWordEncoder(Encoder):
    """
        Core encoder is a stack of N layers.

        Features:
            [1] Use word_embeds to generate word_scores;
            [2] sent embeds: avg (sd) or usee word reps * attn (wsd).

    """
    def __init__(self, layer, n_layers, ins_attn, ins_attn_layer, d_embed):
        super(StefanosWordEncoder, self).__init__(layer, n_layers, ins_attn, ins_attn_layer, d_embed)

    def encode_3d_embeds(self, embed_tensor, mask_mat):
        """
            embed_tensor could be
                word_embed_tensor: n_sents * n_words * d_embed
                OR
                batched_sent_embed_tensor, d_batch * n_sents * d_embed

            mask_mat, 2d mat, could be
                word_mask_mat: n_sents * n_words
                OR
                batched_sent_mask_mat: d_batch * n_sents
        """
        n_ele_in_batch = torch.sum(mask_mat, dim=-1, keepdim=True)  # d_batch, 1. double.
        n_ele_in_batch[n_ele_in_batch == float(0)] = float(1)  # for div, replacing 0 with 1 does not matter

        higher_level_embed_mat = torch.sum(embed_tensor, dim=-2)  # d_batch * d_embed
        denom = torch.sqrt(n_ele_in_batch).expand_as(higher_level_embed_mat).float()  # d_batch, 1 => d_batch, d_embed
        higher_level_embed_mat = higher_level_embed_mat / denom

        mask_mat = mask_mat.unsqueeze(-1)  # 3d mat, n_sents * n_words * 1 or n_batch * n_sents * 1

        for layer in self.layers:
            embed_tensor, _ = layer(embed_tensor, mask_mat)  # just use embed_tensor and attn at the last layer

        if self.norm:
            embed_tensor = self.norm(embed_tensor)

        # print('higher_level_embed_mat: {}'.format(higher_level_embed_mat))
        enc_res = {
            'higher_level_embed_mat': higher_level_embed_mat,
        }

        if self.ins_attn:
            attn_mat = self.build_ins_attn(embed_tensor, mask_mat)  # n_sents * n_words
            enc_res['attn_mat'] = attn_mat

            embed_tensor = embed_tensor.transpose(1, 2).contiguous()  # n_sents * d_embed * n_words
            attn_mat = attn_mat.unsqueeze(-1)  # n_sents * n_words * 1
            higher_level_embed_mat = torch.matmul(embed_tensor, attn_mat).squeeze(-1)  # n_sents * n_dom
            enc_res['higher_level_embed_mat'] = higher_level_embed_mat

        return enc_res

    def forward(self, word_embeds, word_mask):
        """
            word_embeds: d_batch * n_sents * n_words * d_embed or 1 * n_doms * n_sents * n_words * d_embed
            word_mask: d_batch * n_sents * n_words or 1 * n_doms * n_sents * n_words
        """
        sent_embeds_list = list()
        if self.ins_attn:
            word_attn_list = list()

        for word_embed_tensor, mask_mat in zip(word_embeds, word_mask):
            # treat n_sent as d_batch in the loop
            enc_res = self.encode_3d_embeds(word_embed_tensor, mask_mat)

            # word_reps_list.append(enc_res['embed_tensor'])
            sent_embeds_list.append(enc_res['higher_level_embed_mat'])

            if self.ins_attn:
                word_attn_list.append(enc_res['attn_mat'])

        # word_reps = torch.stack(word_reps_list)
        sent_embeds = torch.stack(sent_embeds_list)

        word_enc_res = {
            'sent_embeds': sent_embeds,  # d_batch * n_sents * d_embed
        }

        if self.ins_attn:
            word_enc_res['word_attn'] = torch.stack(word_attn_list)  # d_batch * n_sents * n_words

        return word_enc_res
