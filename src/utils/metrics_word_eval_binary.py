import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from os.path import dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from utils.config_loader import config_model, logger
from utils.tools import save_numpy, extract_dom_scores_from_score_mat
import utils.metrics as metrics


def binarize_y_pred(y_pred, y_true, mode):
    """
        turn y_pred into binary array per the given truth.
        e.g., when the two exact matches,
                y_pred = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                true_labels = [0, 1] (derived from y_true)
            then return positive class [1].

            When there is no exact matching but at least one hit, binary results rely on the mode.
            If mode == 'RELAXED', return positive class;
            if mode == 'STRICT', return negative class.

    :param y_pred: [n_ways,]
    :param true_labels: [n_ways,]
    :param mode: 'RELAXED' | 'STRICT'
    :return: np.array([1]) or np.array([0])
    """
    if y_true.shape != y_pred.shape:
        raise ValueError('Incompatible shape between y_true: {} and y_pred: {}'.format(y_true.shape, y_pred.shape))

    if mode not in ('RELAXED', 'STRICT'):
        raise ValueError('Invalid mode: {}'.format(mode))

    true_labels = [dom_id for dom_id in range(len(y_true)) if y_true[dom_id] == 1.0]
    pred_labels = [dom_id for dom_id in range(len(y_true)) if y_pred[dom_id] == 1.0]

    pos_class = np.array([1], dtype=np.int32)
    neg_class = np.array([0], dtype=np.int32)

    matching = None
    if true_labels == pred_labels:
        matching = 'EXACT'
    else:
        for label in true_labels:
            if label in pred_labels:
                matching = 'ONE_HIT'

    # logger.info('true_labels: {}, pred_labels: {}, matching: {}'.format(true_labels, pred_labels, matching))

    if matching == 'EXACT' or (mode == 'RELAXED' and matching == 'ONE_HIT'):
        return pos_class
    else:
        return neg_class


def binarize_y_pred_3d(y_pred_sents, y_true_sents, mode):
    """
        Binarize sentence predictions per true sentence labels.

        y_pred_sents [batch * max_n_sents * n_ways] =>
        binary_y_pred_3d [batch * max_n_sents * 2]

    :param y_pred_sents: batch * max_n_sents * n_ways
    :param y_true_sents: batch * max_n_sents * n_ways
    :param mode: 'RELAX' | 'STRICT'

    :return: binary_y_pred_sents: batch * max_n_sents * 2
    """
    if y_true_sents.shape != y_pred_sents.shape:
        raise ValueError(
            'Incompatible shape between y_true: {} and y_pred: {}'.format(y_true_sents.shape, y_pred_sents.shape))

    d_batch, max_n_sents = y_true_sents.shape[:-1]
    binary_mats = list()
    for doc_idx in range(d_batch):
        binary_arrs = list()
        for sent_idx in range(max_n_sents):
            y_pred = y_pred_sents[doc_idx, sent_idx, :]
            y_true = y_true_sents[doc_idx, sent_idx, :]
            binary_arr = binarize_y_pred(y_pred=y_pred, y_true=y_true, mode=mode)
            binary_arrs.append(binary_arr)

        binary_mat = np.stack(binary_arrs)
        binary_mats.append(binary_mat)

    binary_y_pred_2d = np.stack(binary_mats)
    return binary_y_pred_2d


def rank_word_score_2d(word_score_mat, y_true_sent, y_true_words, nw, mode, fixed_n_selection=None):
    if y_true_sent.shape != word_score_mat[0].shape:
        raise ValueError(
            'Incompatible shape between y_true: {} and word_score_mat: {}'.format(y_true_sent.shape,
                                                                                  word_score_mat.shape))

    if not nw:  # padded sentence
        return None

    ref_ids_sent = [dom_id for dom_id in range(len(y_true_sent)) if y_true_sent[dom_id] == 1.0]

    if fixed_n_selection:
        n_selection = fixed_n_selection
    else:
        n_selection = len([w_id for w_id in range(len(y_true_words)) if y_true_words[w_id] == 1])

    _, word_dom_scores = extract_dom_scores_from_score_mat(score_mat=word_score_mat, y_ref_ids=ref_ids_sent,
                                                           mode=mode.lower())

    if mode == 'ADD':
        word_items = list(enumerate(word_dom_scores[0][:nw]))
        word_items = sorted(word_items, key=lambda word_item: word_item[1], reverse=True)
        pred_labels = [item[0] for item in word_items][:n_selection]
    elif mode == 'SEP':
        word_items = [item for dom_selection in word_dom_scores for item in list(enumerate(dom_selection[:nw]))]
        word_items = sorted(word_items, key=lambda word_item: word_item[1], reverse=True)
        ranked_labels = [item[0] for item in word_items]
        pred_labels = list()
        for label in ranked_labels:
            if len(pred_labels) == n_selection:
                break

            if label not in pred_labels:
                pred_labels.append(label)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    if nw >= n_selection and len(pred_labels) != n_selection:
        raise ValueError(
            'The size of pred_labels: {0} does not satisfy the expected value of {1}'.format(pred_labels,
                                                                                             n_selection))

    return pred_labels



def rank_and_binarize_word_score_2d(word_score_mat, y_true_sent, y_true_words, nw, mode, fixed_n_selection=None):
    """
        turn y_pred into binary array per the given truth.
        e.g., when the two exact matches,
                y_pred = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                true_labels = [0, 1] (derived from y_true)
            then return positive class [1].

            When there is no exact matching but at least one hit, binary results rely on the mode.
            If mode == 'RELAXED', return positive class;
            if mode == 'STRICT', return negative class.

    :param word_score_mat: [max_n_words, n_ways]
    :param y_true_sent: [n_ways,]
    :param y_true_words: [max_n_words, 1]: for getting the size of word selections
    :param mode: 'ADD' | 'SEP'
    :return: binary_vec: [max_n_words, 1]
    """
    max_n_words = word_score_mat.shape[0]
    binary_vec = np.zeros([max_n_words, 1], dtype=np.int32)

    pred_labels = rank_word_score_2d(word_score_mat, y_true_sent, y_true_words, nw, mode, fixed_n_selection)

    if not pred_labels:
        return binary_vec

    for label in pred_labels:
        binary_vec[label] = 1

    return binary_vec



def binarize_y_pred_4d(y_pred_words, y_true_sents, mode, max_alter=False, word_score_4d=None, n_words=None):
    """
        Binarize word predictions per true sentence labels.

        y_pred_words [batch * max_n_sents * max_n_words * n_ways] =>
        y_pred [batch * max_n_sents * max_n_words]

    :param y_pred_words: batch * max_n_sents * max_n_words * n_ways
    :param y_true_sents: batch * max_n_sents * n_ways

    :return: binary_y_pred_words: batch * max_n_sents * max_n_words * 2
    """
    d_batch, max_n_sents, max_n_words = y_pred_words.shape[:-1]

    if max_alter:
        if word_score_4d is None or n_words is None:
            raise ValueError(
                'word_score_4d: {0} or n_words: {1} should not be None when using max alter!'.format(
                    word_score_4d, n_words))

    binary_tensors = list()
    for doc_idx in range(d_batch):
        binary_mats = list()
        for sent_idx in range(max_n_sents):
            binary_arrs = list()
            y_true = y_true_sents[doc_idx, sent_idx]

            for word_idx in range(max_n_words):
                y_pred = y_pred_words[doc_idx, sent_idx, word_idx, :]
                binary_arr = binarize_y_pred(y_pred=y_pred, y_true=y_true, mode=mode)
                binary_arrs.append(binary_arr)

            binary_mat = np.stack(binary_arrs)

            # select the word with the highest domain score when there is no word selected
            if max_alter:
                word_score_mat = word_score_4d[doc_idx, sent_idx]
                nw = n_words[doc_idx, sent_idx]

                if nw and not binary_mat[:nw].any() == 1:
                    fixed_n_selection = 5
                    binary_mat = rank_and_binarize_word_score_2d(word_score_mat, y_true, y_true_words=None, nw=nw,
                                                                 mode='ADD', fixed_n_selection=fixed_n_selection)
                    logger.info('Make {0} choice for {1}.{2}'.format(fixed_n_selection, doc_idx, sent_idx))

            binary_mats.append(binary_mat)

        binary_tensor = np.stack(binary_mats)
        binary_tensors.append(binary_tensor)

    binary_y_pred_4d = np.stack(binary_tensors)
    return binary_y_pred_4d


def tile_doc_preds_to_sent_preds(y_pred_2d, max_n_sents):
    """

    :param y_pred_2d: d_batch * n_doms
    :param max_n_sents: scala
    :return: y_true_sents: d_batch * max_n_sents * n_doms
    """
    d_batch = len(y_pred_2d)
    y_pred_sents = list()
    # logger.info('n_sents: {}'.format(n_sents))

    for doc_idx in range(d_batch):
        y_pred_sent = np.tile(y_pred_2d[doc_idx, :], (max_n_sents, 1))
        y_pred_sents.append(y_pred_sent)

    y_pred_sents = np.stack(y_pred_sents)
    return y_pred_sents


def tile_sent_preds_to_word_preds(y_pred_3d, max_n_words):
    """

    :param y_pred_3d: d_batch * max_n_sents * 1
    :param max_n_words: scala
    :return: y_true_sents: d_batch * max_n_sents * max_n_words * 1
    """
    d_batch, max_n_sents = y_pred_3d.shape[:2]
    # logger.info('n_sents: {}'.format(n_sents))

    y_pred_words = list()
    for doc_idx in range(d_batch):

        y_pred_words_in_a_doc = list()
        for sent_idx in range(max_n_sents):
            y_pred_w = np.tile(y_pred_3d[doc_idx, sent_idx, :], (max_n_words, 1))
            y_pred_words_in_a_doc.append(y_pred_w)

        y_pred_words_in_a_doc = np.stack(y_pred_words_in_a_doc)
        y_pred_words.append(y_pred_words_in_a_doc)

    y_pred_words = np.stack(y_pred_words)
    return y_pred_words


def compute_exam_p_and_r(y_true_3d, y_pred_3d, n_sents, n_words, silent=False):
    """
        return a list of precision values.

    :param y_true_4d: d_batch * max_n_sents * max_n_words * 1
    :param y_pred_4d: d_batch * max_n_sents * max_n_words * 1
    :param n_sents: d_batch * 1
    :param n_words: d_batch * max_n_sents * 1
    """
    p_list = list()
    r_list = list()
    d_batch = y_true_3d.shape[0]

    for sample_idx in range(d_batch):
        n_sent = n_sents[sample_idx, 0]
        for sent_idx in range(n_sent):
            n_word = n_words[sample_idx, sent_idx]
            y_true = y_true_3d[sample_idx, sent_idx, :n_word]
            y_pred = y_pred_3d[sample_idx, sent_idx, :n_word]
            if not silent and not y_pred.any() == 1:
                logger.info('No pred is made for: {0}.{1}. y_pred: {2}'.format(sample_idx, sent_idx, y_pred))

            p_list.append(precision_score(y_true, y_pred))
            r_list.append(recall_score(y_true, y_pred))

    return p_list, r_list


def get_metric_inputs_wo_flatten(hyp_scores, y_true_sents, y_true_words, n_sents, n_words, pred_grain, max_alter,
                                 matching_mode, binarize_max_alter=True):
    if pred_grain == 'word':
        y_pred_4d = metrics.get_y_pred_4d(hyp_scores, max_alter)
        binary_y_pred_4d = binarize_y_pred_4d(y_pred_4d, y_true_sents, matching_mode, max_alter=binarize_max_alter,
                                              word_score_4d=hyp_scores, n_words=n_words)
        y_true_3d = np.squeeze(y_true_words, axis=-1)
        y_pred_3d = np.squeeze(binary_y_pred_4d, axis=-1)
        return y_true_3d, y_pred_3d

    elif pred_grain in ('sent', 'doc'):
        if pred_grain == 'sent':
            y_pred_3d = metrics.get_y_pred_3d(hyp_scores, max_alter)
        else:  # 'doc'
            y_pred_2d = metrics.get_y_pred_2d(hyp_scores, max_alter)
            y_pred_3d = tile_doc_preds_to_sent_preds(y_pred_2d, max_n_sents=config_model['max_n_sents'])
        binary_y_pred_3d = binarize_y_pred_3d(y_pred_3d, y_true_sents, matching_mode)
        y_pred_4d = tile_sent_preds_to_word_preds(binary_y_pred_3d, max_n_words=config_model['max_n_words'])

        y_true_3d = np.squeeze(y_true_words, axis=-1)
        y_pred_3d = np.squeeze(y_pred_4d, axis=-1)

        return y_true_3d, y_pred_3d

    else:
        raise ValueError('Invalid pred_grain: {}'.format(pred_grain))


def metric_eval_for_mturk_words_with_ir(hyp_scores, y_true_sents, y_true_words, n_sents, n_words, pred_grain, max_alter,
                                        matching_mode, binarize_max_alter=True, save_pred_to=None, save_true_to=None):
    """
        Eval word predictions for mturk.
    """
    para = {
        'hyp_scores': hyp_scores,
        'y_true_sents': y_true_sents,
        'y_true_words': y_true_words,
        'n_sents': n_sents,
        'n_words': n_words,
        'pred_grain': pred_grain,
        'max_alter': max_alter,
        'matching_mode': matching_mode,
        'binarize_max_alter': binarize_max_alter,
    }

    y_true, y_pred = get_metric_inputs_wo_flatten(**para)

    if save_pred_to:
        save_numpy(y_pred, fp=save_pred_to)

    if save_true_to:
        save_numpy(y_true, fp=save_true_to)

    p_list, r_list = compute_exam_p_and_r(y_true, y_pred, n_sents, n_words)

    res = {
        'p_list': p_list,
        'r_list': r_list,
    }

    return res
