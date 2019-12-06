# -*- coding: utf-8 -*-
import numpy as np
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from utils.config_loader import logger, config_model
import data.data_pipe as pipe
import utils.tools as tools
import utils.metrics as metrics
import utils.metrics_word_eval_binary as metrics_word_eval_binary

major_label = tools.get_major_train_doc_label()
y_pred_vec = tools.vectorize_labels(major_label)
max_n_sents = config_model['max_n_sents']


def test_major_doc():
    data_loader = pipe.DomDetDataLoader(dataset_type='test')
    n_iter, total_loss = 0, 0.0
    n_samples, total_hamming = 0, 0.0
    cf_mats, precision_list, recall_list = list(), list(), list()

    for batch_idx, batch in enumerate(data_loader):
        n_iter += 1

        y_true = batch['labels'].cpu().numpy()  # turn vars to numpy arrays
        d_batch = len(y_true)
        y_pred = np.tile(y_pred_vec, (d_batch, 1))

        eval_args = {'y_true': y_true,
                     'hyp_scores': y_pred,
                     }

        n_samples += d_batch
        # logger.info('batch_size: {0}'.format(d_batch))
        eval_res = metrics.metric_eval(**eval_args)

        cf_mats.append(eval_res['cf_mat_list'])
        precision_list.extend(eval_res['precision_list'])
        recall_list.extend(eval_res['recall_list'])
        total_hamming += eval_res['hamming']

    cls_f1, avg_f1 = metrics.compute_f1_with_confusion_mats(cf_mats)
    example_based_f1 = metrics.compute_example_based_f1(precision_list=precision_list, recall_list=recall_list)
    hamming = total_hamming / n_samples

    eval_log_info = {
        'example_based_f1': example_based_f1,
        'avg_f1': avg_f1,
        'cls_f1': cls_f1,
        'hamming': hamming,
    }

    res_str = 'example_based_f1: {example_based_f1:.6f},' \
              'avg_f1: {avg_f1:.6f}, cls_f1: {cls_f1}, hamming: {hamming:.6f}'

    logger.info(res_str.format(**eval_log_info))


def test_major_sent(synthese):
    # logger.info('START: model testing...')
    dataset_type = 'synthese' if synthese else 'lead'
    data_loader = pipe.DomDetDataLoader(dataset_type=dataset_type)

    n_iter, total_loss = 0, 0.0
    n_samples, total_hamming = 0, 0.0

    cf_mats, precision_list, recall_list = list(), list(), list()

    for batch_idx, batch in enumerate(data_loader):
        n_iter += 1

        y_true = batch['labels'].cpu().numpy()
        d_batch = len(y_true)
        des_sent_info = batch['des_sent_info'].cpu().numpy()
        n_samples += np.sum(des_sent_info[:, -1])

        # logger.info('batch_size: {0}'.format(y_true.shape[0]))

        if synthese:
            hyp_scores = np.tile(y_pred_vec, (d_batch, 1))
            fids = batch['fids'].cpu().numpy()
            eval_args = {'hyp_scores': hyp_scores,
                         'fids': fids,
                         'is_hiernet': True
                         }
            eval_res = metrics.metric_eval_for_syn_doc(**eval_args)
        else:
            hyp_scores = np.tile(y_pred_vec, (d_batch, max_n_sents, 1))
            eval_args = {'y_true': y_true,
                         'hyp_scores': hyp_scores,
                         'des_sent_info': des_sent_info,
                         }
            eval_res = metrics.metric_eval(**eval_args)

        cf_mats.append(eval_res['cf_mat_list'])
        precision_list.extend(eval_res['precision_list'])
        recall_list.extend(eval_res['recall_list'])
        total_hamming += eval_res['hamming']

    cls_f1, avg_f1 = metrics.compute_f1_with_confusion_mats(cf_mats)
    example_based_f1 = metrics.compute_example_based_f1(precision_list=precision_list, recall_list=recall_list)
    hamming = total_hamming / n_samples

    eval_log_info = {
        'example_based_f1': example_based_f1,
        'avg_f1': avg_f1,
        'cls_f1': cls_f1,
        'hamming': hamming,
    }

    res_str = 'example_based_f1: {example_based_f1:.6f},' \
              'avg_f1: {avg_f1:.6f}, cls_f1: {cls_f1}, hamming: {hamming:.6f}'

    logger.info(res_str.format(**eval_log_info))


def test_model_sent_mturk():
    logger.info('START: testing Baseline [MAJOR] on [MTURK SENTS]')

    data_loader = pipe.DomDetDataLoader(dataset_type='mturk')

    n_iter, total_loss = 0, 0.0
    n_samples, total_hamming = 0, 0.0
    cf_mats, precision_list, recall_list = list(), list(), list()

    for batch_idx, batch in enumerate(data_loader):
        n_iter += 1

        y_true = batch['sent_labels'].cpu().numpy()  # d_batch * max_n_sents * n_doms
        d_batch = len(y_true)
        hyp_scores = np.tile(y_pred_vec, (d_batch, 1))
        # hyp_scores = np.tile(y_pred_vec, (d_batch, max_n_sents, 1))

        n_sents = batch['n_sents'].cpu().numpy()
        n_samples += np.sum(n_sents)

        logger.info('batch_size: {0}'.format(y_true.shape[0]))

        eval_args = {'y_true': y_true,
                     'hyp_scores': hyp_scores,
                     'n_sents': n_sents,
                     'is_hiernet': True,
                     }
        eval_res = metrics.metric_eval_for_mturk(**eval_args)

        cf_mats.append(eval_res['cf_mat_list'])
        precision_list.extend(eval_res['precision_list'])
        recall_list.extend(eval_res['recall_list'])
        total_hamming += eval_res['hamming']

    cls_f1, avg_f1 = metrics.compute_f1_with_confusion_mats(cf_mats)
    example_based_f1 = metrics.compute_example_based_f1(precision_list=precision_list, recall_list=recall_list)
    hamming = total_hamming / n_samples

    eval_log_info = {
        'example_based_f1': example_based_f1,
        'avg_f1': avg_f1,
        'cls_f1': cls_f1,
        'hamming': hamming,
    }

    res_str = 'example_based_f1: {example_based_f1:.6f},' \
              'avg_f1: {avg_f1:.6f}, cls_f1: {cls_f1}, hamming: {hamming:.6f}'

    logger.info(res_str.format(**eval_log_info))


def test_model_word_mturk(matching_mode=None, corpus='wiki'):
    logger.info('START: model testing on [MTURK WORDS]')

    grain = 'word'
    dataset_type = '-'.join(('mturk', corpus, grain))
    data_loader = pipe.DomDetDataLoader(dataset_type=dataset_type)

    n_samples = 0
    p_list = list()
    r_list = list()

    for batch_idx, batch in enumerate(data_loader):
        # turn vars to numpy arrays
        y_true_sents = batch['sent_labels'].cpu().numpy()  # d_batch * max_n_sents * n_doms
        y_true_words = batch['word_labels'].cpu().numpy()  # d_batch * max_n_sents * max_n_words
        n_sents = batch['n_sents'].cpu().numpy()
        n_words = batch['n_words'].cpu().numpy()
        n_samples += np.sum(n_sents)

        d_batch = len(y_true_sents)
        hyp_scores = np.tile(y_pred_vec, (d_batch, 1))

        logger.info('batch_size: {0}'.format(y_true_words.shape[0]))

        eval_args = {'hyp_scores': hyp_scores,
                     'y_true_sents': y_true_sents,
                     'y_true_words': y_true_words,
                     'n_sents': n_sents,
                     'n_words': n_words,
                     'pred_grain': 'doc',
                     'max_alter': True,
                     'matching_mode': matching_mode,
                     }

        eval_res = metrics_word_eval_binary.metric_eval_for_mturk_words_with_ir(**eval_args)

        p_list.extend(eval_res['p_list'])
        r_list.extend(eval_res['r_list'])

    exam_f1 = metrics.compute_example_based_f1(p_list, r_list)
    logger.info('word-eval. exam_f1: {0:6f}'.format(exam_f1))


if __name__ == '__main__':
    # test_major_doc()
    # test_major_sent(synthese=False)
    # test_major_sent(synthese=True)
    # test_model_sent_mturk()
    test_model_word_mturk(matching_mode='RELAXED', corpus='wiki')
