# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.config_loader import logger


class Detector(nn.Module):
    def __init__(self, d_embed, n_doms, score_func, dropout, activate_func=None, gate=None, embed_layer=None):
        super(Detector, self).__init__()
        self.d_embed = d_embed
        self.n_doms = n_doms
        self.score_func = score_func
        self.dropout = nn.Dropout(dropout)
        self.activate_func = activate_func
        self.gate = gate
        self.embed_layer = embed_layer

    def _embeds_3d_id_tensors(self, word_ids):
        """

        :param word_ids: 3d
        :return: a 4-d tensor.
        """
        # if word_ids.dim() == 4:
        #     word_ids = word_ids.squeeze(0)
        return torch.stack([self.embed_layer(word_id_mat) for word_id_mat in word_ids])

    def activate(self, input):
        assert self.activate_func in ('tanh', 'relu', 'elu')

        if self.activate_func == 'tanh':
            return F.tanh(input)
        elif self.activate_func == 'relu':
            return F.relu(input)
        else:
            return F.elu(input)

    def get_scores(self, score_in):
        assert self.score_func in ('tanh', 'sigmoid', 'softmax')
        score_in = self.dropout(score_in)

        if self.score_func == 'tanh':
            return F.tanh(score_in)
        elif self.score_func == 'sigmoid':
            return F.sigmoid(score_in)
        else:
            return F.softmax(score_in, dim=-1)

    def _bn(self, input, bn, out_size):
        return bn(input.view(-1, self.n_doms)).view(out_size)

    def _compute_sem_scores(self, reps, linear):
        scores = self.get_scores(linear(reps))
        return scores, scores.size()

    def _gate_input(self, dynamic_in, linear, rand_in):
        if self.gate == 'dynamic':
            gate_in = linear(torch.stack(dynamic_in, dim=-1))
        else:
            gate_in = rand_in.expand_as(dynamic_in[-1])

        gate = F.sigmoid(gate_in)  # d_batch * n_sents * n_words * n_dom

        return gate * dynamic_in[-1] + (1 - gate) * dynamic_in[-2]

    def _gate_input_dynamically(self, dynamic_in, linear, bn=None, after_activation=False, gate_ctr=0.1):
        assert self.gate == 'dynamic'

        gate_in = linear(torch.cat(dynamic_in, dim=-1))

        # apply bn before sigmoid to ensure gate elements are in (0, 1)
        if bn and not after_activation:
            d_gate = gate_in.size()
            # gate_in = gate_in.view(-1, self.n_doms)
            gate_in = self._bn(input=gate_in, bn=bn, out_size=d_gate)

        gate = F.sigmoid(gate_in)  # d_batch * n_sents * n_words * n_dom

        if bn and after_activation:
            d_gate = gate.size()
            gate = self._bn(input=gate, bn=bn, out_size=d_gate)

        if gate_ctr != 1.0:
            gate = gate_ctr * gate

        # logger.info('gate: {0}'.format(gate))

        return gate * dynamic_in[-1] + (1 - gate) * dynamic_in[-2]

    def forward(self, *input):
        raise NotImplementedError


class WordDomDetector(Detector):
    """
        two parts:
        (1) words semantics;
        (2) incorporate domain definition as side information (wait to be done).
    """
    def __init__(self, d_embed, n_doms, score_func, dropout, activate_func, side_info, gate, embed_layer, word_enc, sent_enc, use_gate_bn=None):
        super(WordDomDetector, self).__init__(d_embed, n_doms, score_func, dropout, activate_func, gate, embed_layer)
        self.side_info = side_info
        self.W_z = nn.Linear(d_embed, n_doms)
        # uncomment when use bn
        self.sem_bn = nn.BatchNorm1d(n_doms)

        if side_info:
            des_ids = side_info['des_ids']
            des_word_mask = side_info['des_word_mask']
            des_sent_mask = side_info['des_sent_mask']

            self.register_buffer('des_ids', des_ids)
            self.register_buffer('des_word_mask', des_word_mask)
            self.register_buffer('des_sent_mask', des_sent_mask)

            self.word_enc = word_enc
            self.sent_enc = sent_enc

            self.W_u = nn.Linear(d_embed, d_embed)
            # uncomment when use bn
            self.side_inner_bn = nn.BatchNorm1d(d_embed)
            self.side_outer_bn = nn.BatchNorm1d(n_doms)
            self.word_bn = nn.BatchNorm1d(n_doms)

            self.side_gate_bn = nn.BatchNorm1d(n_doms) if use_gate_bn else None

            if self.gate == 'dynamic':
                self.W_g_q = nn.Linear(d_embed+n_doms*2, n_doms)
            elif self.gate == 'rand_vec':
                self.side_gate_in = nn.Parameter(torch.rand(n_doms))
            else:
                self.side_gate_in = nn.Parameter(torch.rand(1))

    def _create_desc_reps(self):
        """
            create desc_reps with desc word ids, word mask and sent mask.

        :param desc_ids: 4d tensor, 1 * n_doms * n_sents * n_words
        :param desc_word_mask: 4d tensor, 1 * n_doms * n_sents * n_words
        :param desc_sent_mask: 3d tensor, 1 * n_doms * n_sents

        :return: desc_reps: 3d tensor, 1 * n_doms * d_embed
        """
        # if config_model['side_dim_expand']:
        #     des_ids = des_ids.squeeze(0)
        #     des_word_mask = des_word_mask.squeeze(0)
        #     des_sent_mask = des_sent_mask.squeeze(0)

        des_ids = Variable(self.des_ids, requires_grad=False)
        des_word_mask = Variable(self.des_word_mask, requires_grad=False)
        des_sent_mask = Variable(self.des_sent_mask, requires_grad=False)

        des_word_embeds = self._embeds_3d_id_tensors(des_ids)

        des_sent_embeds = self.word_enc(des_word_embeds, des_word_mask)['sent_embeds']  # n_doms * n_sents * n_words * d_embed

        des_sent_reps = self.sent_enc(des_sent_embeds, des_sent_mask)['sent_reps']  # n_doms * n_sents * d_embed

        des_reps = torch.sum(des_sent_reps, dim=-2)  # n_doms * d_embed

        # doc mask
        n_dom_sents = torch.sum(des_sent_mask, dim=-1, keepdim=True)  # n_doms * 1
        n_dom_sents[n_dom_sents == float(0)] = 1  # for div, do not matter by replacing 0 with 1
        des_rep_denom = torch.sqrt(n_dom_sents).expand_as(des_reps).float()  # n_doms * d_embed

        des_reps = des_reps / des_rep_denom
        # desc_reps = desc_reps / desc_rep_denom.type(torch.FloatTensor)

        return des_reps

    def _compute_side_scores(self, word_reps, des_reps):
        side_score_in_left = self.side_inner_bn(self.activate(self.W_u(des_reps)))  # n_dom * d_embed
        side_score_in_right = torch.t(word_reps.view(-1, self.d_embed))  # d_embed * -1
        side_score_in = torch.matmul(side_score_in_left, side_score_in_right)  # n_dom * -1

        side_score_in = torch.t(side_score_in).contiguous()  # -1 * n_dom

        return self.get_scores(side_score_in)  # -1 * n_dom

    def forward(self, word_reps):
        """
        :param word_reps: d_batch * n_sents * n_words * d_embed
        :param descs_reps: n_dom * d_embed, if use.

        :return: word_dists: d_batch * n_sents * n_words * n_doms
        """
        sem_scores, d_word_scores = self._compute_sem_scores(word_reps, self.W_z)  # d_batch * n_sents * n_words * n_dom

        # uncomment when use bn
        sem_scores = self._bn(sem_scores, self.sem_bn, d_word_scores)

        if not self.side_info:
            # logger.info('word_scores: {0}'.format(word_scores))
            return sem_scores

        des_reps = self._create_desc_reps()  # n_doms * d_embed

        # logger.info('Done: des_reps: {0}'.format(des_reps.size()))

        side_scores = self._compute_side_scores(word_reps, des_reps)  # -1 * n_dom

        # uncomment when use bn
        side_scores = self._bn(side_scores, self.side_outer_bn, d_word_scores)  # d_batch * n_sents * n_words * n_dom
        # print(side_scores.size())

        # print('Word - side_scores: {0}'.format(side_scores))
        # print('Word - sem_scores: {0}'.format(sem_scores))

        dynamic_in = [word_reps, sem_scores, side_scores]
        # logger.info('Prior gate [WORD]...')
        word_scores = self._gate_input_dynamically(dynamic_in, self.W_g_q, bn=self.side_gate_bn)

        # uncomment when use bn
        word_scores = self._bn(word_scores, self.word_bn, d_word_scores)

        return word_scores


class StefanosWordDomDetector(Detector):
    """
        two parts:
        (1) words semantics;
        (2) incorporate domain definition as side information (wait to be done).
    """
    def __init__(self, d_embed, n_doms, score_func, dropout, activate_func, gate, embed_layer):
        super(StefanosWordDomDetector, self).__init__(d_embed, n_doms, score_func, dropout, activate_func, gate, embed_layer)
        self.W_z = nn.Linear(d_embed, n_doms)
        # uncomment when use bn
        self.sem_bn = nn.BatchNorm1d(n_doms)

    def forward(self, word_embeds):
        """
        :param word_embeds: d_batch * n_sents * n_words * d_embed

        :return: word_scores: d_batch * n_sents * n_words * n_doms
        """
        sem_scores, d_word_scores = self._compute_sem_scores(word_embeds, self.W_z)  # d_batch * n_sents * n_words * n_dom

        # uncomment when use bn
        word_scores = self._bn(sem_scores, self.sem_bn, d_word_scores)

        return word_scores


class SentDomDetector(Detector):
    """
        combine word_dists as instances with word-level self-attention.
    """
    def __init__(self, d_embed, n_doms, score_func, dropout, activate_func, gate, embed_layer, use_mil, use_gate_bn=None):
        super(SentDomDetector, self).__init__(d_embed, n_doms, score_func, dropout, activate_func, gate, embed_layer)
        self.use_mil = use_mil
        # uncomment when use bn
        self.sem_bn = nn.BatchNorm1d(n_doms)
        self.W_h = nn.Linear(d_embed, n_doms)

        if use_mil:
            # uncomment when use bn
            self.mil_bn = nn.BatchNorm1d(n_doms)
            self.sent_bn = nn.BatchNorm1d(n_doms)

            self.upward_gate_bn = nn.BatchNorm1d(n_doms) if use_gate_bn else None

            if gate == 'dynamic':
                self.W_g = nn.Linear(d_embed+n_doms*2, n_doms)
            elif gate == 'rand_vec':
                self.upward_gate_in = nn.Parameter(torch.rand(n_doms))
            else:
                self.upward_gate_in = nn.Parameter(torch.rand(1))

    def _compute_ins_scores(self, word_scores, word_attn):
        word_scores = word_scores.transpose(-2, -1).contiguous()  # d_batch * n_sents * n_dom * n_words
        d_word_scores = list(word_scores.size())
        n_words = d_word_scores[-1]
        word_scores = word_scores.view(-1, self.n_doms, n_words) # (d_batch * n_sents) * n_dom * n_words

        instance_attn = word_attn.unsqueeze(-1).view(-1, n_words, 1)  # (d_batch * n_sents) * n_words * 1
        ins_scores = torch.matmul(word_scores, instance_attn).squeeze(-1)  # (d_batch * n_sents) * n_dom
        # print('Sent - mil_scores: {0}'.format(mil_scores))
        # logger.info('lambda_p: {0}, side_scores: {1}, mil_scores: {2}'.format(lambda_p.type, side_scores.type, mil_scores.type))
        return ins_scores

    def forward(self, sent_reps, word_scores=None, word_attn=None):
        """
        :param word_scores: d_batch * n_sents * n_words * n_doms
        :param word_attn: # d_batch * n_sents * n_words
        :param sent_reps: d_batch * n_sents * d_embed, has been transposed.

        :return: sent_dists: d_batch * n_sents * n_doms
        """
        if self.use_mil and (word_scores is None or word_attn is None):
            raise ValueError('cannot use mil while there is no instance or its importance.')

        sem_scores, d_sent_scores = self._compute_sem_scores(sent_reps, self.W_h)  # d_batch * n_sents * n_doms
        # uncomment when use bn
        sem_scores = self._bn(input=sem_scores, bn=self.sem_bn, out_size=d_sent_scores)  # d_batch * n_sents * n_doms

        res_dict = {
                'sem_scores': sem_scores
            }

        if not self.use_mil:
            res_dict['sent_scores'] = sem_scores
            return res_dict

        if self.use_mil:
            ins_scores = self._compute_ins_scores(word_scores, word_attn)
            # uncomment when use bn
            ins_scores = self._bn(input=ins_scores, bn=self.mil_bn, out_size=d_sent_scores)  # d_batch * n_sents * n_doms
            # compose with upward gate
            dynamic_in = [sent_reps, sem_scores, ins_scores]

            # logger.info('Upward gate ...')
            sent_scores = self._gate_input_dynamically(dynamic_in, linear=self.W_g, bn=self.upward_gate_bn)
            # uncomment when use bn
            sent_scores = self._bn(input=sent_scores, bn=self.sent_bn, out_size=d_sent_scores)

            res_dict['sent_scores'] = sent_scores
            return res_dict


class StefanosSentDomDetector(Detector):
    """
        detect sent_scores from sent-level semantics and
        (optionally) combining word_scores as instances with word-level self-attention.
    """
    def __init__(self, d_embed, n_doms, score_func, dropout, activate_func, gate, embed_layer, use_mil, use_gate_bn=None):
        super(StefanosSentDomDetector, self).__init__(d_embed, n_doms, score_func, dropout, activate_func, gate, embed_layer)
        self.use_mil = use_mil
        # uncomment when use bn
        self.sem_bn = nn.BatchNorm1d(n_doms)
        self.W_h = nn.Linear(d_embed, n_doms)

        if use_mil:
            # uncomment when use bn
            self.mil_bn = nn.BatchNorm1d(n_doms)
            self.sent_bn = nn.BatchNorm1d(n_doms)

            self.upward_gate_bn = nn.BatchNorm1d(n_doms) if use_gate_bn else None

            if gate == 'dynamic':
                self.W_g = nn.Linear(d_embed+n_doms*2, n_doms)
            elif gate == 'rand_vec':
                self.upward_gate_in = nn.Parameter(torch.rand(n_doms))
            else:
                self.upward_gate_in = nn.Parameter(torch.rand(1))

    def _compute_ins_scores(self, word_scores, word_attn):
        word_scores = word_scores.transpose(-2, -1).contiguous()  # d_batch * n_sents * n_dom * n_words
        d_word_scores = list(word_scores.size())
        n_words = d_word_scores[-1]
        word_scores = word_scores.view(-1, self.n_doms, n_words) # (d_batch * n_sents) * n_dom * n_words

        instance_attn = word_attn.unsqueeze(-1).view(-1, n_words, 1)  # (d_batch * n_sents) * n_words * 1
        ins_scores = torch.matmul(word_scores, instance_attn).squeeze(-1)  # (d_batch * n_sents) * n_dom
        # print('Sent - mil_scores: {0}'.format(mil_scores))
        # logger.info('lambda_p: {0}, side_scores: {1}, mil_scores: {2}'.format(lambda_p.type, side_scores.type, mil_scores.type))
        return ins_scores

    def forward(self, sent_embeds, word_scores=None, word_attn=None):
        """
        :param sent_embeds: d_batch * n_sents * d_embed, has been transposed.
        :param word_scores: d_batch * n_sents * n_words * n_doms
        :param word_attn: # d_batch * n_sents * n_words
        :return: sent_scores: d_batch * n_sents * n_doms
        """

        if self.use_mil and (word_scores is None or word_attn is None):
            raise ValueError('cannot use mil while there is no instance or its importance.')

        sem_scores, d_sent_scores = self._compute_sem_scores(sent_embeds, self.W_h)  # d_batch * n_sents * n_doms
        # uncomment when use bn
        sem_scores = self._bn(input=sem_scores, bn=self.sem_bn, out_size=d_sent_scores)  # d_batch * n_sents * n_doms

        if not self.use_mil:
            return sem_scores

        ins_scores = self._compute_ins_scores(word_scores, word_attn)
        # uncomment when use bn
        ins_scores = self._bn(input=ins_scores, bn=self.mil_bn, out_size=d_sent_scores) # d_batch * n_sents * n_doms
        # compose with upward gate
        dynamic_in = [sent_embeds, sem_scores, ins_scores]

        # logger.info('Upward gate ...')
        sent_scores = self._gate_input_dynamically(dynamic_in, linear=self.W_g, bn=self.upward_gate_bn)
        # uncomment when use bn
        sent_scores = self._bn(input=sent_scores, bn=self.sent_bn, out_size=d_sent_scores)

        return sent_scores


class DocDomDetector(nn.Module):
    """
        combine sent_dists as instances with sent-level self-attention.
        include sent-level self-attention.
    """
    def __init__(self, n_doms, dropout=0.1):
        super(DocDomDetector, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(n_doms)

    def forward(self, sent_scores, sent_attn):
        """

        :param sent_scores: batch * n_sents * n_dom
        :param sent_attn: d_batch * n_sents
        :return: doc_scores: batch * n_dom
        """
        sent_scores = sent_scores.transpose(1, 2)  # batch * n_dom * n_sents
        sent_attn = sent_attn.unsqueeze(-1)  # batch * n_sents * 1

        doc_scores = torch.matmul(sent_scores, sent_attn).squeeze(-1)  # batch * n_sents

        # doc_scores = F.tanh(doc_scores)
        doc_scores = self.bn(doc_scores)

        # logger.info('doc_scores: {0}'.format(doc_scores))

        return doc_scores


class DocDomDetectorWithReps(Detector):
    """
        combine sent_dists as instances with sent-level self-attention.
        include sent-level self-attention.
    """
    def __init__(self, d_embed, n_doms, score_func, dropout, activate_func):
        super(DocDomDetectorWithReps, self).__init__(d_embed, n_doms, score_func, dropout, activate_func)
        self.dropout = nn.Dropout(dropout)
        self.W_d = nn.Linear(d_embed, n_doms)
        self.bn = nn.BatchNorm1d(n_doms)

    def archived_forward_wo_attn(self, sent_reps):
        """

        :param sent_reps: batch * n_sents * d_embed
        :return: doc_scores: batch * n_dom
        """
        doc_reps = torch.sum(sent_reps, dim=1)  # batch * d_embed

        doc_scores_in = self.W_d(doc_reps)
        doc_scores = self.get_scores(doc_scores_in)
        doc_scores = self.bn(doc_scores)

        return doc_scores

    def forward(self, sent_reps, sent_attn):
        """

        :param sent_reps: batch * n_sents * d_embed
        :param sent_attn: d_batch * n_sents
        :return: doc_scores: batch * n_dom
        """
        # logger.info('sent_reps: {0}'.format(sent_reps))
        # logger.info('sent_attn: {0}'.format(sent_attn))

        # logger.info('sent_attn: {0}'.format(sent_attn))

        sent_reps = sent_reps.transpose(1, 2)  # batch * d_embed * n_sents
        sent_attn = sent_attn.unsqueeze(-1)  # batch * n_sents * 1

        doc_reps = torch.matmul(sent_reps, sent_attn).squeeze(-1)   # batch * d_embed
        # logger.info('doc_reps: {0}'.format(doc_reps))

        doc_scores_in = self.W_d(doc_reps)
        doc_scores = self.get_scores(doc_scores_in)

        # logger.info('doc_scores (before bn): {0}'.format(doc_scores))

        doc_scores = self.bn(doc_scores)

        # logger.info('doc_scores: {0}'.format(doc_scores))

        return doc_scores
