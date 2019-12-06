# -*- coding: utf-8 -*-
import io
import numpy as np
from os.path import dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from frame.checkpoint_op import *
import data.data_pipe as pipe
import utils.metrics as metrics
import utils.config_loader as config_loader
from utils.config_loader import path_parser, lang, doms_final
from utils.tools import build_res_str, get_model_pred_grain
import utils.metrics_word_eval_binary as metrics_word_eval_binary


def test_model_word_mturk(model, matching_mode, corpus='wiki', save_pred=False, save_gold=False, restore=False):
    checkpoint = join(path_parser.model_save, config_loader.model_name)
    if restore:
        checkpoint = join(checkpoint, 'resume')
    fns = [fn.split('.')[0] for fn in listdir(checkpoint) if fn.endswith('.pth.tar')]
    for n_iter in fns:
        logger.info('===============================')
        test_model_word_mturk_with_checkpoints(model, matching_mode, corpus, save_pred, save_gold, n_iter, restore)


def test_model_word_mturk_with_checkpoints(model, matching_mode=None, corpus='wiki', save_pred=False,
                                           save_gold=False, n_iter=None, restore=False):
    if corpus == 'wiki':
        save_dir = path_parser.pred_mturk_wiki
    elif corpus == 'nyt':
        if lang != 'en':
            raise ValueError('Set lang to en when NYT corpus is used')
        save_dir = path_parser.pred_mturk_nyt
    else:
        raise ValueError('Invalid corpus: {}'.format(corpus))

    if config_loader.placement == 'auto':
        model = nn.DataParallel(model, device_ids=config_loader.device)

    if config_loader.placement in ('auto', 'single'):
        model.cuda()

    logger.info('START: model testing on [MTURK WORDS]')

    checkpoint = join(path_parser.model_save, config_loader.model_name)
    if restore:
        checkpoint = join(checkpoint, 'resume')

    filter_keys = None
    if config_loader.reset_size_for_test and not config_loader.set_sep_des_size:
        logger.info('Filter DES pretrained paras...')
        filter_keys = ['module.word_det.des_ids', 'module.word_det.des_sent_mask', 'module.word_det.des_word_mask']
    load_checkpoint(checkpoint=checkpoint, model=model, n_iter=n_iter, filter_keys=filter_keys)

    grain = 'word'
    dataset_type = '-'.join(('mturk', corpus, grain))
    data_loader = pipe.DomDetDataLoader(dataset_type=dataset_type)
    model.eval()

    c = copy.deepcopy
    pred_grain = get_model_pred_grain()
    p_list = list()
    r_list = list()
    y_true_sents_list = list()
    n_sents_list = list()

    for batch_idx, batch in enumerate(data_loader):

        feed_dict = c(batch)

        del feed_dict['sent_labels']
        del feed_dict['word_labels']
        del feed_dict['n_sents']
        del feed_dict['n_words']

        for (k, v) in feed_dict.items():
            feed_dict[k] = Variable(v, requires_grad=False, volatile=True)  # fix ids and masks

        if pred_grain == 'doc':
            _, doc_scores = model(**feed_dict)
            hyp_scores = doc_scores.data.cpu().numpy()
        elif pred_grain == 'sent':
            _, _, sent_scores = model(**feed_dict)
            hyp_scores = sent_scores.data.cpu().numpy()
        elif pred_grain == 'word':
            feed_dict['return_sent_attn'] = True
            feed_dict['return_word_attn'] = True
            _, _, _, word_scores, _, word_attn = model(**feed_dict)

            hyp_scores = word_scores.data.cpu().numpy()  # n_batch * n_sents * n_words * n_doms
        else:
            raise ValueError('Invalid prediction grain: {}'.format(pred_grain))

        # turn vars to numpy arrays
        y_true_sents = batch['sent_labels'].cpu().numpy()  # d_batch * max_n_sents * n_doms
        y_true_words = batch['word_labels'].cpu().numpy()  # d_batch * max_n_sents * max_n_words
        n_sents = batch['n_sents'].cpu().numpy()
        n_words = batch['n_words'].cpu().numpy()

        logger.info('batch_size: {0}'.format(y_true_words.shape[0]))

        eval_args = {'hyp_scores': hyp_scores,
                     'y_true_sents': y_true_sents,
                     'y_true_words': y_true_words,
                     'n_sents': n_sents,
                     'n_words': n_words,
                     'pred_grain': pred_grain,
                     'max_alter': True,
                     'matching_mode': matching_mode,
                     }

        if save_pred:
            fn = '_'.join((grain, config_loader.meta_model_name))
            pred_save_fp = join(save_dir, fn)
            eval_args['save_pred_to'] = pred_save_fp

        if save_gold:
            fn = '_'.join((grain, 'gold'))
            true_save_fp = join(save_dir, fn)
            eval_args['save_true_to'] = true_save_fp

        eval_res = metrics_word_eval_binary.metric_eval_for_mturk_words_with_ir(**eval_args)

        p_list.extend(eval_res['p_list'])
        r_list.extend(eval_res['r_list'])
        y_true_sents_list.append(y_true_sents)
        n_sents_list.append(n_sents)

    exam_f1 = metrics.compute_example_based_f1(p_list, r_list)
    logger.info('word-eval. exam_f1: {0:6f}'.format(exam_f1))

    report_dom_specific_f1(p_list, r_list, y_true_sents_list[0], n_sents_list[0])


def report_dom_specific_f1(p_list, r_list, y_true_sents, n_sents):
    dom2f1 = dict()
    f1s = list()
    for p, r in zip(p_list, r_list):
        if p + r:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0
        f1s.append(f1)

    if len(f1s) != np.sum(n_sents):
        raise ValueError('Incompatible f1s. #f1s: {0} while #sents: {1}'.format(len(f1s), np.sum(n_sents)))

    d_batch = y_true_sents.shape[0]

    count = 0
    for doc_id in range(d_batch):
        ns = n_sents[doc_id, 0]
        if not ns:
            break
        for sent_id in range(ns):
            y_true = y_true_sents[doc_id, sent_id]
            dom_ids = [dom_id for dom_id in range(len(y_true)) if y_true[dom_id] == 1.0]
            for dom_id in dom_ids:
                dom = doms_final[dom_id]
                f1 = f1s[count]
                if dom_id not in dom2f1:
                    dom2f1[dom] = [f1]
                else:
                    dom2f1[dom].append(f1)
            count += 1

    for dom, f1_list in dom2f1.items():
        f1 = float(sum(f1_list)) / len(f1_list)
        logger.info('{0}: f1: {1:6f}'.format(dom, f1))


def select_pos_neg_labels_for_words(word_score_4d, y_true_sents, y_true_words, n_words, mode, fixed_n_selection=None):
    d_batch, max_n_sents, max_n_words = word_score_4d.shape[:-1]

    records = list()
    for doc_idx in range(d_batch):
        for sent_idx in range(max_n_sents):
            nw = n_words[doc_idx, sent_idx]
            pos_w_ids = metrics_word_eval_binary.rank_word_score_2d(word_score_mat=word_score_4d[doc_idx, sent_idx, :],
                                                                    y_true_sent=y_true_sents[doc_idx, sent_idx],
                                                                    y_true_words=y_true_words[doc_idx, sent_idx],
                                                                    nw=nw,
                                                                    mode=mode,
                                                                    fixed_n_selection=fixed_n_selection)
            if not pos_w_ids:
                continue

            pos_w_ids = [str(w_id) for w_id in pos_w_ids]
            pos_str = '|'.join(pos_w_ids)
            neg_w_ids = [str(w_id) for w_id in range(nw) if str(w_id) not in pos_w_ids]
            neg_str = '|'.join(neg_w_ids)

            record = '\t'.join((str(doc_idx), str(sent_idx), pos_str, neg_str))
            records.append(record)

    return records


def load_id2wids(corpus):
    if lang == 'en':
        if corpus == 'wiki':
            sample_fp = path_parser.sampled_label_en_wiki
        elif corpus == 'nyt':
            sample_fp = path_parser.sampled_label_en_nyt
        else:
            raise ValueError('Invalid corpus with EN: {}'.format(corpus))
    else:
        if corpus == 'wiki':
            sample_fp = path_parser.sampled_label_zh_wiki
        else:
            raise ValueError('Invalid corpus with ZH: {}'.format(corpus))

    with io.open(sample_fp, encoding='utf-8') as sample_f:
        lines = sample_f.readlines()

    id2pos = dict()
    id2neg = dict()

    for line in lines:
        items = line.rstrip('\n').split('\t')
        doc_id, sent_id, true_pos, true_neg = items
        id = '.'.join((doc_id, sent_id))
        id2pos[id] = true_pos
        id2neg[id] = true_neg

    logger.info('Load {} sentences for pos and neg word eval, respectively'.format(len(id2pos)))

    return id2pos, id2neg
