# -*- coding: utf-8 -*-
from utils.config_loader import device, placement
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import data.data_pipe as pipe
from utils.config_loader import config_model, n_ways
from frame.encoder import *
from frame.detector import *
from frame.transformer import *


class ModelBase(nn.Module):
    def __init__(self, embed_layer, word_enc, sent_enc, doc_det, loss_layer, sent_det=None):
        super(ModelBase, self).__init__()
        self.embed_layer = embed_layer
        self.word_enc = word_enc
        self.sent_enc = sent_enc
        self.sent_det = sent_det
        self.doc_det = doc_det
        self.loss_layer = loss_layer
        self.gru_div = None

    def _embeds_3d_id_tensors(self, word_ids):
        """

        :param word_ids: 3d
        :return: a 4-d tensor.
        """
        # if word_ids.dim() == 4:
        #     word_ids = word_ids.squeeze(0)
        return torch.stack([self.embed_layer(word_id_mat) for word_id_mat in word_ids])

    def _create_word_embeds(self, word_ids):
        return self._embeds_3d_id_tensors(word_ids)

    def _compute_loss(self, doc_scores, ref):
        # d_batch = hyp.size()[0]
        # loss = self.loss_layer(hyp, ref) / d_batch
        # return loss
        return self.loss_layer(doc_scores, ref)

    def forward(self, *input):
        raise NotImplementedError


class DetNet(ModelBase):
    use_des = True
    use_sent_mil = True
    use_doc_mil = True

    def __init__(self, embed_layer, word_enc, sent_enc, word_det, sent_det, doc_det, loss_layer):
        super(DetNet, self).__init__(embed_layer, word_enc, sent_enc, doc_det, loss_layer, sent_det)
        self.word_det = word_det

    def forward(self, labels, word_ids, word_mask, sent_mask, return_sent_attn=False, return_word_attn=False):
        """

        :param labels: d_batch * n_doms
        :param word_ids: d_batch * n_sents * n_words
        :param word_mask: d_batch * n_sents * n_words
        :param sent_mask: d_batch * n_sents
        :return: doc_scores: 2d mat, d_batch * n_dom
        """
        # logger.info('Start: enc - word')
        word_embeds = self._create_word_embeds(word_ids)
        word_enc_res = self.word_enc(word_embeds, word_mask)
        word_reps, word_attn, sent_embeds = word_enc_res['word_reps'], word_enc_res['word_attn'], word_enc_res['sent_embeds']

        # logger.info('Start: enc - sent')
        sent_enc_res = self.sent_enc(sent_embeds, sent_mask)
        sent_reps, sent_attn = sent_enc_res['sent_reps'], sent_enc_res['sent_attn']

        # logger.info('Start: det - word')
        word_scores = self.word_det(word_reps)

        # det - sent
        # logger.info('Start: det - sent')
        sent_det_paras = {
            'word_scores': word_scores,
            'word_attn': word_attn,
            'sent_reps': sent_reps,
        }

        # logger.info('Done: sent_reps: {0}'.format(sent_reps))
        # sent_scores = self.sent_det(**sent_det_paras)
        sent_res_dict = self.sent_det(**sent_det_paras)
        # sem_scores = sent_res_dict['sem_scores']
        sent_scores = sent_res_dict['sent_scores']

        # det - doc
        # logger.info('Start: det - doc')
        doc_scores = self.doc_det(sent_scores, sent_attn)  # d_batch * n_dom
        # logger.info('Done: doc_scores: {}'.format(doc_scores))

        loss = self._compute_loss(doc_scores=doc_scores, ref=labels)

        if return_sent_attn and return_word_attn:
            return loss, doc_scores, sent_scores, word_scores, sent_attn, word_attn

        if return_sent_attn:
            return loss, doc_scores, sent_scores, sent_attn

        if return_word_attn:
            return loss, doc_scores, sent_scores, word_scores, word_attn

        return loss, doc_scores, sent_scores, word_scores


class DetNet2(ModelBase):
    use_sent_mil = True
    use_doc_mil = True
    use_des = False

    def __init__(self, embed_layer, word_enc, sent_enc, word_det, sent_det, doc_det, loss_layer):
        super(DetNet2, self).__init__(embed_layer, word_enc, sent_enc, doc_det, loss_layer, sent_det)
        self.word_det = word_det

    def forward(self, labels, word_ids, word_mask, sent_mask, return_sent_attn=False, return_word_attn=False):
        """

        :param labels: d_batch * n_doms
        :param word_ids: d_batch * n_sents * n_words
        :param word_mask: d_batch * n_sents * n_words
        :param sent_mask: d_batch * n_sents

        :return: doc_scores: 2d mat, d_batch * n_dom
        """
        # enc - word
        # logger.info('Start: enc - word')
        word_embeds = self._create_word_embeds(word_ids)
        word_enc_res = self.word_enc(word_embeds, word_mask)
        word_reps, word_attn, sent_embeds = word_enc_res['word_reps'], word_enc_res['word_attn'], word_enc_res['sent_embeds']

        # logger.info('Start: enc - sent')
        sent_enc_res = self.sent_enc(sent_embeds, sent_mask)
        sent_reps, sent_attn = sent_enc_res['sent_reps'], sent_enc_res['sent_attn']

        # logger.info('Start: det - word')
        word_scores = self.word_det(word_reps)

        # logger.info('Start: det - sent')
        sent_det_paras = {
            'word_scores': word_scores,
            'word_attn': word_attn,
            'sent_reps': sent_reps,
        }

        sent_res_dict = self.sent_det(**sent_det_paras)
        sem_scores = sent_res_dict['sem_scores']
        sent_scores = sent_res_dict['sent_scores']

        # logger.info('Start: det - doc')
        doc_scores = self.doc_det(sent_scores, sent_attn)  # d_batch * n_dom

        loss = self._compute_loss(doc_scores=doc_scores, ref=labels)

        if return_sent_attn and return_word_attn:
            return loss, doc_scores, sent_scores, word_scores, sent_attn, word_attn

        if return_sent_attn:
            return loss, doc_scores, sent_scores, sent_attn

        if return_word_attn:
            return loss, doc_scores, sent_scores, word_scores, word_attn

        return loss, doc_scores, sent_scores, word_scores


class DetNet1(ModelBase):
    use_des = False
    use_sent_mil = False
    use_doc_mil = True

    def __init__(self, embed_layer, word_enc, sent_enc, sent_det, doc_det, loss_layer):
        if placement == 'manual':
            embed_layer = embed_layer.cuda(device[0])

            word_enc = word_enc.cuda(device[1])

            sent_enc = sent_enc.cuda(device[-1])
            sent_det = sent_det.cuda(device[-1])
            doc_det = doc_det.cuda(device[-1])
            loss_layer = loss_layer.cuda(device[-1])

        super(DetNet1, self).__init__(embed_layer, word_enc, sent_enc, doc_det, loss_layer, sent_det)

    def forward(self, labels, word_ids, word_mask, sent_mask, return_sent_attn=False):
        """

        :param labels: d_batch * n_doms
        :param word_ids: d_batch * n_sents * n_words
        :param word_mask: d_batch * n_sents * n_words
        :param sent_mask: d_batch * n_sents

        :return: doc_scores: 2d mat, d_batch * n_dom
        """
        if placement == 'manual':
            word_ids = word_ids.cuda(device[0])

        # logger.info('Start: enc - word')
        word_embeds = self._create_word_embeds(word_ids)

        if placement == 'manual':
             word_embeds = word_embeds.cuda(device[1])
             word_mask = word_mask.cuda(device[1])

        word_enc_res = self.word_enc(word_embeds, word_mask)
        sent_embeds = word_enc_res['sent_embeds']

        if placement == 'manual':
            # logger.info('Start: P2P GPU transfer...')
            sent_embeds = sent_embeds.cuda(device[-1])
            sent_mask = sent_mask.cuda(device[-1])
            labels = labels.cuda(device[-1])

        # logger.info('Start: enc - sent')
        sent_enc_res = self.sent_enc(sent_embeds, sent_mask)
        sent_reps, sent_attn = sent_enc_res['sent_reps'], sent_enc_res['sent_attn']

        # logger.info('Start: det - sent')

        # sent_scores = self.sent_det(sent_reps=sent_reps)
        sent_res_dict = self.sent_det(sent_reps=sent_reps)
        sent_scores = sent_res_dict['sent_scores']

        # logger.info('Start: det - doc')
        doc_scores = self.doc_det(sent_scores, sent_attn)  # d_batch * n_dom

        loss = self._compute_loss(doc_scores=doc_scores, ref=labels)

        if return_sent_attn:
            return loss, doc_scores, sent_scores, sent_attn

        return loss, doc_scores, sent_scores


class HierNet(ModelBase):
    use_sent_mil = False
    use_doc_mil = False
    use_des = False

    def __init__(self, embed_layer, word_enc, sent_enc, doc_det, loss_layer):
        super(HierNet, self).__init__(embed_layer, word_enc, sent_enc, doc_det, loss_layer)

    def forward(self, labels, word_ids, word_mask, sent_mask):
        """

        :param labels: d_batch * n_doms
        :param word_ids: d_batch * n_sents * n_words
        :param word_mask: d_batch * n_sents * n_words
        :param sent_mask: d_batch * n_sents
        :return: doc_scores: 2d mat, d_batch * n_dom
        """
        word_embeds = self._create_word_embeds(word_ids)
        # logger.info('Start: enc - word')
        word_enc_res = self.word_enc(word_embeds, word_mask)
        sent_embeds = word_enc_res['sent_embeds']

        # logger.info('Start: enc - sent')
        sent_enc_res = self.sent_enc(sent_embeds, sent_mask)
        sent_reps, sent_attn = sent_enc_res['sent_reps'], sent_enc_res['sent_attn']

        # logger.info('Start: det - doc')
        doc_scores = self.doc_det(sent_reps, sent_attn)  # d_batch * n_dom

        loss = self._compute_loss(doc_scores=doc_scores, ref=labels)

        return loss, doc_scores


class MILNet(ModelBase):
    """
        context-free MIL applied to sent-doc.

        word_reps=>sent_embed: average with no attn;
        sent_scores=>doc_scores: key attn.
    """
    use_sent_mil = False
    use_doc_mil = True
    use_des = False
    use_top = False

    ws_attn = False
    sd_attn = 'key'

    def __init__(self, embed_layer, word_enc, sent_enc, sent_det, doc_det, loss_layer):
        super(MILNet, self).__init__(embed_layer, word_enc, sent_enc, doc_det, loss_layer, sent_det)

    def forward(self, labels, word_ids, word_mask, sent_mask, return_sent_attn=False):
        """

        :param labels: d_batch * n_doms
        :param word_ids: d_batch * n_sents * n_words
        :param word_mask: d_batch * n_sents * n_words
        :param sent_mask: d_batch * n_sents

        :return: doc_scores: 2d mat, d_batch * n_dom
        """
        # logger.info('Start: enc - word')
        word_embeds = self._create_word_embeds(word_ids)

        word_enc_res = self.word_enc(word_embeds, word_mask)
        sent_embeds = word_enc_res['sent_embeds']

        # logger.info('Start: enc - sent')
        sent_enc_res = self.sent_enc(sent_embeds, sent_mask)
        sent_attn = sent_enc_res['sent_attn']

        # logger.info('Start: det - sent')
        sent_scores = self.sent_det(sent_embeds=sent_embeds)

        # logger.info('Start: det - doc')
        doc_scores = self.doc_det(sent_scores, sent_attn)  # d_batch * n_dom

        loss = self._compute_loss(doc_scores=doc_scores, ref=labels)


        if return_sent_attn:
            return loss, doc_scores, sent_scores, sent_attn

        return loss, doc_scores, sent_scores


def make_prior(use_des, use_top):
    """
    :param model: could be model class or model instance.
    :return: prior required for the input model type.
    """
    if not (use_des or use_top):
        return None

    return pipe.PriorLoader(transform=pipe.ToTensor()).load_prior(use_des=use_des, use_top=use_top)


def make_loss_layer(score_func):
    if score_func == 'tanh':
        return nn.SoftMarginLoss()
    else:
        return nn.MultiLabelSoftMarginLoss()

    # class SoftmaxLossLayer(nn.Module):
    #     def __init__(self):
    #         super(SoftmaxLossLayer, self).__init__()
    #         self.nll = nn.NLLLoss()
    #
    #     def forward(self, doc_scores, target):
    #         return self.nll(torch.log(doc_scores + 1e-7), target)
    #
    # return SoftmaxLossLayer()


def make_opt(model):
    if config_model['opt'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=config_model['lr'], weight_decay=config_model['weight_decay'])
    else:
        raise ValueError('Opt in model config should be adam!')


def make_detnet_model(vocab_size, use_sent_embed_attn, use_gate_bn=True):
    """
        Helper: Construct a model from hyperparameters.
        :use_gate_bn: batch norm for gate scores
        :use_sent_embed_attn: if set to True, use attn to combine word reps => sent embeds
        :return model and its save name (constructed by key paras).
    """
    d_embed = config_model['d_embed']
    d_model = config_model['d_model']
    dropout = config_model['dropout']

    model_cls = globals()[config_model['variation']]
    use_des, use_top = model_cls.use_des, model_cls.use_top
    use_sent_mil, use_doc_mil = model_cls.use_sent_mil, model_cls.use_doc_mil

    c = copy.deepcopy
    attn = MultiHeadedAttention(h=config_model['n_heads'], d_model=d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, config_model['d_ff'], dropout)
    position = PositionalEncoding(d_model, dropout)
    embed_layer = nn.Sequential(Embeddings(d_model, vocab_size), c(position))

    word_ins_attn, word_ins_attn_layer = None, None

    if use_sent_mil or use_sent_embed_attn:
        word_ins_attn = config_model['ins_attn']
        if word_ins_attn == 'multi':
            word_ins_attn_layer = InsAttnLayer(d_embed, c(attn))

    # for model with use_doc_mil=None,
    # this sent attn combines sent reps (instead of sent scores) for doc reps.
    sent_ins_attn = config_model['ins_attn']
    sent_ins_attn_layer = None
    if sent_ins_attn == 'multi':
        sent_ins_attn_layer = InsAttnLayer(d_embed, c(attn))

    word_enc_paras = {
        'layer': EncoderLayer(d_model, c(attn), c(ff), dropout),
        'n_layers': config_model['n_layers'],
        'ins_attn': word_ins_attn,
        'ins_attn_layer': word_ins_attn_layer,
        'd_embed': d_embed,
    }

    sent_enc_paras = {
        'layer': EncoderLayer(d_model, c(attn), c(ff), dropout),
        'n_layers': config_model['n_layers'],
        'ins_attn': sent_ins_attn,
        'ins_attn_layer': sent_ins_attn_layer,
        'd_embed': d_embed,
        'pos_enc': c(position),
    }

    if use_sent_embed_attn:
        WordEncCls = WordEncoderWithAttnRep
    else:
        WordEncCls = WordEncoder
    word_enc = WordEncCls(**word_enc_paras)
    sent_enc = SentEncoder(**sent_enc_paras)

    loss_layer = make_loss_layer(config_model['score_func'])

    model_components = {
        'embed_layer': embed_layer,
        'word_enc': word_enc,
        'sent_enc': sent_enc,
        'loss_layer': loss_layer,
    }

    side_info = make_prior(use_des, use_top)

    basic_det_paras = {
        'd_embed': d_embed,
        'n_doms': n_ways,
        'score_func': config_model['score_func'],
        'dropout': dropout,
        'activate_func': config_model['activate_func'],
    }

    if use_sent_mil:
        side_des = c(side_info) if use_des else None
        word_det_paras = {**basic_det_paras,
                          'side_info': side_des,
                          'gate': config_model['gate'],
                          'embed_layer': embed_layer,
                          'word_enc': word_enc,
                          'sent_enc': sent_enc,
                          'use_gate_bn': use_gate_bn,
                          }
        model_components['word_det'] = WordDomDetector(**word_det_paras)

    if use_doc_mil:
        side_top = c(side_info) if use_top else None
        sent_det_para = {**basic_det_paras,
                         'side_info': side_top,
                         'gate': config_model['gate'],
                         'embed_layer': embed_layer,
                         'use_mil': use_sent_mil,
                         'use_gate_bn': use_gate_bn,
                         }
        model_components['sent_det'] = SentDomDetector(**sent_det_para)
        # model_components['sent_det'] = SentDomDetectorwoUpwardGate(**sent_det_para)
        model_components['doc_det'] = DocDomDetector(n_doms=n_ways, dropout=dropout)
    else:
        model_components['doc_det'] = DocDomDetectorWithReps(**basic_det_paras)

    model = model_cls(**model_components)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
            # nn.init.kaiming_uniform(p)

    return model


def make_milnet_model(vocab_size, use_gate_bn=True):
    d_embed = config_model['d_embed']
    d_model = config_model['d_model']
    dropout = config_model['dropout']

    c = copy.deepcopy
    attn = MultiHeadedAttention(h=config_model['n_heads'], d_model=d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, config_model['d_ff'], dropout)
    position = PositionalEncoding(d_model, dropout)
    embed_layer = nn.Sequential(Embeddings(d_model, vocab_size), c(position))

    model_cls = globals()[config_model['variation']]
    use_sent_mil, use_doc_mil = model_cls.use_sent_mil, model_cls.use_doc_mil
    ws_attn, sd_attn= model_cls.ws_attn, model_cls.sd_attn

    word_ins_attn_layer, sent_ins_attn_layer = None, None
    if ws_attn == 'multi':
        word_ins_attn_layer = InsAttnLayer(d_embed, c(attn))
    if sd_attn == 'multi':
        sent_ins_attn_layer = InsAttnLayer(d_embed, c(attn))

    word_enc_paras = {
        'layer': EncoderLayer(d_model, c(attn), c(ff), dropout),
        'n_layers': config_model['n_layers'],
        'ins_attn': ws_attn,
        'ins_attn_layer': word_ins_attn_layer,
        'd_embed': d_embed,
    }

    sent_enc_paras = {
        'layer': EncoderLayer(d_model, c(attn), c(ff), dropout),
        'n_layers': config_model['n_layers'],
        'ins_attn': sd_attn,
        'ins_attn_layer': sent_ins_attn_layer,
        'd_embed': d_embed,
        'pos_enc': c(position),
    }

    word_enc = StefanosWordEncoder(**word_enc_paras)
    sent_enc = SentEncoder(**sent_enc_paras)

    loss_layer = make_loss_layer(config_model['score_func'])

    model_components = {
        'embed_layer': embed_layer,
        'word_enc': word_enc,
        'sent_enc': sent_enc,
        'loss_layer': loss_layer,
    }

    basic_det_paras = {
        'd_embed': d_embed,
        'n_doms': n_ways,
        'score_func': config_model['score_func'],
        'dropout': dropout,
        'activate_func': config_model['activate_func'],
    }

    if use_sent_mil:
        word_det_paras = {**basic_det_paras,
                          'gate': config_model['gate'],
                          'embed_layer': embed_layer,
                          }
        model_components['word_det'] = StefanosWordDomDetector(**word_det_paras)

    sent_det_para = {**basic_det_paras,
                     'gate': config_model['gate'],
                     'embed_layer': embed_layer,
                     'use_mil': use_sent_mil,
                     'use_gate_bn': use_gate_bn,
                     }

    model_components['sent_det'] = StefanosSentDomDetector(**sent_det_para)
    model_components['doc_det'] = DocDomDetector(n_doms=n_ways, dropout=dropout)

    model = model_cls(**model_components)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
