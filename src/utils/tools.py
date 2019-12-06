import io
import pickle
import numpy as np
from os import listdir, mkdir
from os.path import exists, join, dirname, abspath, isfile

import re
import itertools
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from utils.config_loader import logger, path_parser, doms_final, config_model, n_ways
from sklearn.metrics import cohen_kappa_score
import utils.config_loader as config_loader

def save_obj(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def get_all_words(fp):
    SENT_PATTERN = '#s-sent[\s\S]*?(?=#e-sent)'  # positive lookahead
    with io.open(fp, encoding='utf-8') as f:
        pattern = re.compile(SENT_PATTERN)
        sents = re.findall(pattern, f.read())

    if not sents:
        logger.warning('no sents found...')
        return None

    # the first line of a sent is #s-sent and is discarded
    sents = [[word_line.rstrip('\n').split('\t')[0]
              for word_line in sent.split('\n')[1:]]
             for sent in sents]

    words = list(itertools.chain.from_iterable(sents))

    return words


def get_fns(dataset_type):
    assert dataset_type in ('train', 'dev', 'test', 'test_filtered')
    if config_loader.mode == 'debug':
        dataset_type = 'debug'

    root = path_parser.dataset_type_dp[dataset_type]
    fns = [fn for fn in listdir(root) if isfile(join(root, fn))]

    return root, fns


def vectorize_labels(labels, zero_as_no=False):
    """
        zero_as_no: for lda, set irrelevant labels as 0.
    """
    if type(labels) == str:
        labels = [labels]

    if config_model['score_func'] == 'tanh' and not zero_as_no:
        label_vec = - np.ones([n_ways], dtype=np.float32)
    else:
        label_vec = np.zeros([n_ways], dtype=np.float32)

    for label in labels:
        label_vec[doms_final.index(label)] = 1.0

    return label_vec


def get_labels(fn, label_format='vec', zero_as_no=False):
    """
        return labels for a doc.
        format: 'vec', 'list', or 'str'
        zero_as_no: for lda, set irrelevant labels as 0.
    """
    labels = fn.split('_')[1:]

    if label_format == 'list':
        return labels

    if label_format == 'str':
        return '_'.join(labels)

    return vectorize_labels(labels, zero_as_no=zero_as_no)


def get_file_id(fn):
    id = int(fn.split('_')[0])
    return np.array([id])


def get_n_sents(label_fn):
    with io.open(join(path_parser.dataset_syn_docs, label_fn), encoding='utf-8') as f:
        list_of_labels = [line.strip('\n').split('_') for line in f.readlines()]  # todo: check the last \n line

    return len(list_of_labels)


def get_major_train_doc_label():
    dict_label_num = dict()
    train_fns = [fn for fn in listdir(path_parser.dataset_train) if isfile(join(path_parser.dataset_train, fn))]
    for fn in train_fns:
        label_str = get_labels(fn, label_format='str')
        if label_str not in dict_label_num:
            dict_label_num[label_str] = 1
        else:
            dict_label_num[label_str] += 1

    sorted_label_num = sorted(dict_label_num.items(), key=lambda item: item[1], reverse=True)

    major_label = sorted_label_num[0][0]

    return major_label


def compute_kappa_for_lead_sent_asm(annot_fp):
    with open(annot_fp, encoding='utf-8') as f:
        file_chunks = f.read().split('==========\n')
        file_chunks = [chunk for chunk in file_chunks if chunk]
        assert len(file_chunks) == 20

        doc_label_list = list()
        des_label_list = list()

        for chunk in file_chunks:
            lines = [line for line in chunk.split('\n') if line]
            fn, n_sents = lines[0], int(lines[1])
            doc_label_vec = get_labels(fn)
            doc_label_tilde_mat = np.tile(doc_label_vec, (n_sents, 1))
            doc_label_list.append(doc_label_tilde_mat)

            if len(lines) > 2:
                diff_lines = lines[2:]
                for diff_line in diff_lines:
                    des_label_vec = - np.ones([len(doms_final)], dtype=np.float32)
                    actual_labels = diff_line.split('\t')
                    for label in actual_labels:
                        des_label_vec[doms_final.index(label)] = 1.0

                    des_label_vec = des_label_vec.reshape((1, -1))
                    des_label_list.append(des_label_vec)

            padding = np.tile(doc_label_vec, (n_sents - len(lines) + 2, 1))
            if padding.ndim == 1:
                padding = padding.reshape((1, -1))
            des_label_list.append(padding)

        doc_label_mat = np.concatenate(doc_label_list)
        des_label_mat = np.concatenate(des_label_list)
        assert doc_label_mat.shape == des_label_mat.shape

    kappa_list = list()
    for cls_idx in range(n_ways):
        kappa_list.append(cohen_kappa_score(des_label_mat[:, cls_idx], doc_label_mat[:, cls_idx]))

    return kappa_list


def compute_avg_kappa_for_lead_sent_asm():
    kappa_list_1 = compute_kappa_for_lead_sent_asm(annot_fp=path_parser.lca_annotation)
    kappa_list_2 = compute_kappa_for_lead_sent_asm(annot_fp=path_parser.lca_annotation_2)

    avg_kappa_1 = sum(kappa_list_1) / n_ways
    avg_kappa_2 = sum(kappa_list_2) / n_ways
    avg_kappa = (avg_kappa_1 + avg_kappa_2) / 2

    logger.info('avg kappa: {0}, avg_kappa_1: {1}, avg_kappa_2: {2}'.format(avg_kappa, avg_kappa_1, avg_kappa_2))


def build_res_str(stage=None, use_loss=True, use_exam_f1=True, use_hamming=True):
    """
        build results string for logging.
    :param stage: ep, iter, or None
    :return:
    """
    prefix = '\t{ph}: '
    if stage:
        add_stuff = '{' + '{stage}'.format(stage=stage) + '} '
        prefix += add_stuff

    main_items = list()

    if use_loss:
        main_items.append('loss: {loss:.6f}')

    if use_exam_f1:
        main_items.append('example_based_f1: {example_based_f1:.6f}')

    main_items.append('avg_f1: {avg_f1:.6f}, cls_f1: {cls_f1}')

    if use_hamming:
        main_items.append('hamming: {hamming:.6f}')

    main_str = ', '.join(main_items)
    res_str = prefix + main_str

    return res_str


def get_model_pred_grain():
    name = config_loader.meta_model_name
    if name == 'hiernet':
        return 'doc'
    elif name in ('detnet1', 'milnet'):
        return 'sent'
    elif name in ('detnet2', 'detnet'):
        return 'word'
    else:
        return ValueError('Invalid model name: {}'.format(name))


def save_numpy(mat, fp):
    str_list = [' '.join([str(scalar) for scalar in vec]) for vec in mat.tolist()]

    with io.open(fp, mode='a', encoding='utf-8') as f:
        f.write('\n'.join(str_list) + '\n')
        # logger.info('save to: {0}'.format(fp))


def extract_dom_scores_from_score_mat(score_mat, y_ref_ids, mode):
    """
        extract target domain scores from a score matrix,
        e.g. sentence scores in a doc, or word scores in a sentence.

    :param score_mat: sent dom scores for all label ids
    :param y_ref_ids: given label ids
    :param mode: 'sep', 'add', 'sep_and_add'
    :return:
    """
    assert mode in ('sep', 'add', 'sep_and_add')

    # scores = list()
    # for label_id in y_ref_ids:
    #     scores.append(score_mat[:, label_id])
    if not y_ref_ids:
        raise ValueError('y_ref_ids can not be empty!')

    scores = [score_mat[:, label_id] for label_id in y_ref_ids]

    y_ref_ids = [str(label_id) for label_id in y_ref_ids]

    if mode == 'sep' or len(y_ref_ids) == 1:
        return y_ref_ids, scores

    concat_ids = '_'.join(y_ref_ids)
    sum_scores = np.sum(scores, axis=0)

    if mode == 'add':
        return [concat_ids], [sum_scores]

    y_ref_ids.append(concat_ids)
    scores.append(sum_scores)

    # logger.info('y_ref_ids: {}'.format(y_ref_ids))

    return y_ref_ids, scores


def get_numerical_doc_id(fn):
    id = int(fn.split('_')[0])
    return np.array([id])


def get_str_doc_id(fn):
    return fn.split('_')[0]


def get_doc_id(fn):
    if config_loader.lang == 'en':
        return get_numerical_doc_id(fn)
    else:
        return get_str_doc_id(fn)


def get_wiki_summ_rank_fp(accurate_only):
    if accurate_only:
        save_fn = '{}_accurate'.format(config_model['variation'])
    else:
        save_fn = '{}_all'.format(config_model['variation'])

    summ_rank_fp = join(path_parser.summary_rank_dps['wiki'], save_fn)
    return summ_rank_fp


def get_wiki_summ_text_dp(order_subdir, accurate_only, multi_label_only, variation=None):
    if not variation:
        variation = config_model['variation']
    text_root = join(path_parser.summary_text_dps['wiki'], variation)

    if not exists(text_root):
        logger.info("Saving root does not exist. Making directory {}".format(text_root))
        mkdir(text_root)

    if accurate_only:
        subdir = '{}_accurate'.format(order_subdir)
    else:
        subdir = '{}_all'.format(order_subdir)

    if multi_label_only:
        subdir = '{}_multilabel'.format(subdir)

    summ_text_dp = join(text_root, subdir)
    if not exists(summ_text_dp):
        logger.info("Saving root does not exist. Making directory {}".format(summ_text_dp))
        mkdir(summ_text_dp)

    return summ_text_dp


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



if __name__ == '__main__':
    compute_avg_kappa_for_lead_sent_asm()
