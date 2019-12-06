# -*- coding: utf-8 -*-
import numpy as np
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from frame.checkpoint_op import *
import data.data_pipe as pipe
import utils.metrics as metrics
import utils.config_loader as config_loader
from utils.config_loader import path_parser, lang
from utils.tools import build_res_str


def test_model_sent(model, save_pred=False, save_gold=False, restore=False):
    checkpoint = join(path_parser.model_save, config_loader.model_name)
    if restore:
        checkpoint = join(checkpoint, 'resume')
    fns = [fn.split('.')[0] for fn in listdir(checkpoint) if fn.endswith('.pth.tar')]
    for n_iter in fns:
        logger.info('===============================')
        test_model_sent_syn_with_checkpoints(model, save_pred, save_gold, n_iter, restore)


def test_model_sent_mturk(model, corpus='wiki', save_pred=False, save_gold=False, restore=False):
    checkpoint = join(path_parser.model_save, config_loader.model_name)
    if restore:
        checkpoint = join(checkpoint, 'resume')
    fns = [fn.split('.')[0] for fn in listdir(checkpoint) if fn.endswith('.pth.tar')]
    for n_iter in fns:
        logger.info('===============================')
        test_model_sent_mturk_with_checkpoints(model, corpus, save_pred, save_gold, n_iter, restore)


def test_model_sent_syn_with_checkpoints(model, save_pred=False, save_gold=False, n_iter=None, restore=False):
    if config_loader.placement == 'auto':
        model = nn.DataParallel(model, device_ids=config_loader.device)

    if config_loader.placement in ('auto', 'single'):
        model.cuda()

    logger.info('START: model testing on [SENTS with SYNTHETIC CONTEXT]')

    checkpoint = join(path_parser.model_save, config_loader.model_name)
    if restore:
        checkpoint = join(checkpoint, 'resume')

    filter_keys = None
    if config_loader.reset_size_for_test and not config_loader.set_sep_des_size:
        logger.info('Filter DES pretrained paras...')
        filter_keys = ['module.word_det.des_ids', 'module.word_det.des_sent_mask', 'module.word_det.des_word_mask']
    load_checkpoint(checkpoint=checkpoint, model=model, n_iter=n_iter, filter_keys=filter_keys)

    data_loader = pipe.DomDetDataLoader(dataset_type='synthese')
    model.eval()

    n_iter, total_loss = 0, 0.0
    n_samples, total_hamming = 0, 0.0

    cf_mats, precision_list, recall_list = list(), list(), list()

    is_hiernet = True if config_loader.meta_model_name == 'hiernet' else False
    if config_loader.meta_model_name in ('detnet1', 'milnet'):
        no_word_scores = True
    else:
        no_word_scores = False

    for batch_idx, batch in enumerate(data_loader):
        n_iter += 1

        c = copy.deepcopy
        feed_dict = c(batch)

        # del paras not for forward()
        del feed_dict['des_sent_info']
        del feed_dict['fids']

        for (k, v) in feed_dict.items():
            feed_dict[k] = Variable(v, requires_grad=False, volatile=True)  # fix ids and masks

        if is_hiernet:
            loss, doc_scores = model(**feed_dict)
            hyp_scores = doc_scores.data.cpu().numpy()
        elif no_word_scores:
            loss, doc_scores, sent_scores = model(**feed_dict)
            hyp_scores = sent_scores.data.cpu().numpy()
        else:
            loss, doc_scores, sent_scores, word_scores = model(**feed_dict)
            hyp_scores = sent_scores.data.cpu().numpy()

        total_loss += loss.data[0]
        # turn vars to numpy arrays
        y_true = batch['labels'].cpu().numpy()
        des_sent_info = batch['des_sent_info'].cpu().numpy()
        n_samples += np.sum(des_sent_info[:, -1])

        fids = batch['fids'].cpu().numpy()
        eval_args = {'hyp_scores': hyp_scores,
                     'fids': fids,
                     'is_hiernet': is_hiernet,
                     }

        if save_pred:
            pred_save_fp = join(path_parser.pred_syn, config_loader.meta_model_name)
            eval_args['save_pred_to'] = pred_save_fp

        if save_gold:
            true_save_fp = join(path_parser.pred_syn, 'gold')
            eval_args['save_true_to'] = true_save_fp

        eval_res = metrics.metric_eval_for_syn_doc(**eval_args)

        cf_mats.append(eval_res['cf_mat_list'])
        precision_list.extend(eval_res['precision_list'])
        recall_list.extend(eval_res['recall_list'])
        total_hamming += eval_res['hamming']

    cls_f1, avg_f1 = metrics.compute_f1_with_confusion_mats(cf_mats)

    eval_log_info = {
        'ph': 'Test',
        'avg_f1': avg_f1,
        'cls_f1': cls_f1,
    }

    res_str = build_res_str(stage=None, use_loss=False, use_exam_f1=False, use_hamming=False)

    logger.info(res_str.format(**eval_log_info))


def test_model_sent_mturk_with_checkpoints(model, corpus='wiki', save_pred=False, save_gold=False, n_iter=None, restore=False):
    if corpus == 'nyt' and lang != 'en':
        raise ('Set lang to en when NYT corpus is used')

    if config_loader.placement == 'auto':
        model = nn.DataParallel(model, device_ids=config_loader.device)

    if config_loader.placement in ('auto', 'single'):
        model.cuda()

    logger.info('START: model testing on [SENTS with MTURK]')

    checkpoint = join(path_parser.model_save, config_loader.model_name)
    if restore:
        checkpoint = join(checkpoint, 'resume')

    filter_keys = None
    if config_loader.reset_size_for_test and not config_loader.set_sep_des_size:
        logger.info('Filter DES pretrained paras...')
        filter_keys = ['module.word_det.des_ids', 'module.word_det.des_sent_mask', 'module.word_det.des_word_mask']
    load_checkpoint(checkpoint=checkpoint, model=model, n_iter=n_iter, filter_keys=filter_keys)

    grain = 'sent'
    dataset_type = '-'.join(('mturk', corpus, grain))
    data_loader = pipe.DomDetDataLoader(dataset_type=dataset_type)
    model.eval()

    n_iter, total_loss = 0, 0.0
    n_samples = 0

    cf_mats, precision_list, recall_list = list(), list(), list()

    is_hiernet = True if config_loader.meta_model_name == 'hiernet' else False
    if config_loader.meta_model_name in ('detnet1', 'milnet'):
        no_word_scores = True
    else:
        no_word_scores = False

    for batch_idx, batch in enumerate(data_loader):
        n_iter += 1

        c = copy.deepcopy
        feed_dict = c(batch)

        del feed_dict['n_sents']
        del feed_dict['sent_labels']
        for (k, v) in feed_dict.items():
            feed_dict[k] = Variable(v, requires_grad=False, volatile=True)  # fix ids and masks

        if is_hiernet:
            loss, doc_scores = model(**feed_dict)
            hyp_scores = doc_scores.data.cpu().numpy()
        elif no_word_scores:
            loss, doc_scores, sent_scores = model(**feed_dict)
            hyp_scores = sent_scores.data.cpu().numpy()
        else:
            loss, doc_scores, sent_scores, word_scores = model(**feed_dict)
            hyp_scores = sent_scores.data.cpu().numpy()

            # for doc_id, score in enumerate(hyp_scores):
            #     for sent_id, s in enumerate(score):
            #         logger.info('{0}.{1}: {2}'.format(doc_id, sent_id, score))

        total_loss += loss.data[0]
        # turn vars to numpy arrays
        y_true = batch['sent_labels'].cpu().numpy()  # d_batch * max_n_sents * n_doms
        n_sents = batch['n_sents'].cpu().numpy()
        # logger.info('n_sents: {0}'.format(n_sents))
        n_samples += np.sum(n_sents)

        # logger.info('batch_size: {0}'.format(y_true.shape[0]))

        eval_args = {'y_true': y_true,
                     'hyp_scores': hyp_scores,
                     'n_sents': n_sents,
                     'is_hiernet': is_hiernet,
                     }

        save_dir = path_parser.pred_mturk_wiki if corpus == 'wiki' else path_parser.pred_mturk_nyt
        if save_pred:
            pred_save_fp = join(save_dir, config_loader.meta_model_name)
            eval_args['save_pred_to'] = pred_save_fp

        if save_gold:
            true_save_fp = join(save_dir, 'gold')
            eval_args['save_true_to'] = true_save_fp

        eval_res = metrics.metric_eval_for_mturk(**eval_args)

        cf_mats.append(eval_res['cf_mat_list'])
        precision_list.extend(eval_res['precision_list'])
        recall_list.extend(eval_res['recall_list'])

    cls_f1, avg_f1 = metrics.compute_f1_with_confusion_mats(cf_mats)

    eval_log_info = {
        'ph': 'Test',
        'avg_f1': avg_f1,
        'cls_f1': cls_f1,
    }

    res_str = build_res_str(stage=None, use_loss=False, use_exam_f1=False, use_hamming=False)
    logger.info(res_str.format(**eval_log_info))
