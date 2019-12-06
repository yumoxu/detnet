import csv
import io
from os.path import join
import itertools
import numpy as np

from utils.config_loader import path_parser, logger, config_model, lang, n_ways
from data.dataset_parser import dataset_parser

class DocRec:
    def __init__(self):
        """
            Init a doc record.

            self.sent_annots and self.sent_labels are domain annotations and labels for sentences;
                they store label indices.

            self.word_annots and self.word_labels are domain annotations and labels for words.
                they store word indices.
        """
        self.sent_annots = list()  # n_worker * max_n_sents * n_labels
        self.sent_labels = list()  # max_n_sents * n_labels

        self.word_annots = list()  # n_worker * max_n_sents * n_labels
        self.word_labels = list()  # max_n_sents * max_n_words * 2

    def __str__(self):
        doc_headline = '===============fn: {0}\tn_sents: {1}==============='.format(self.fn, self.n_sents)
        iter_triples = zip(self.sent_labels, self.word_labels, self.sents)
        sent_strs = ['{0}\t{1}\t{2}'.format(s_label, w_label, sent) for s_label, w_label, sent in iter_triples]
        sent_strs.insert(0, doc_headline)
        sent_str = '\n'.join(sent_strs)

        sent_annot_headline = '===============sentence annotations==============='
        annot_list = ['\t'.join(['|'.join(annot) for annot in annots]) for annots in self.sent_annots]
        annot_list.insert(0, sent_annot_headline)
        sent_annot_str = '\n'.join(annot_list)

        word_annot_headline = '===============word annotations==============='
        word_annot_list = ['\t'.join(['|'.join(annot) for annot in annots]) for annots in self.word_annots]
        word_annot_list.insert(0, word_annot_headline)
        word_annot_str = '\n'.join(word_annot_list)

        return '\n'.join((sent_str, sent_annot_str, word_annot_str))

    def set_fn(self, fn):
        self.fn = fn

    def set_n_sents(self, n_sents):
        if type(n_sents) is not int:
            n_sents = int(n_sents)
        self.n_sents = n_sents
        # print('set_n_sents: {0}'.format(n_sents))

    # def set_n_words(self, n_word_list):
    #     """
    #         Set self.n_words vec.
    #
    #     :param n_words: [1, 2, 3]
    #     """
    #     n_words = np.zeros([config_model['max_n_sents'], 1])
    #     n_words[:len(n_word_list)] = n_word_list
    #     self.n_words = n_words

    def set_sents(self, sents):
        self.sents = sents

    def add_sent_annots(self, annots):
        self.sent_annots.append(annots)

    def add_word_annots(self, annots):
        self.word_annots.append(annots)

    def select_sent_labels(self, threshold):
        if self.sent_labels:
            raise ValueError('sent_labels has already existed: {}'.format(self.sent_labels))

        for sent_idx in range(self.n_sents):
            sent_labels = [annots[sent_idx] for annots in self.sent_annots]

            for item in sent_labels:
                if len(item) == n_ways:  # not eligible annotation
                    sent_labels.remove(item)

            merged = [int(item) for item in list(itertools.chain(*sent_labels)) if item]

            count = [merged.count(idx) for idx in range(n_ways)]

            # add "c == max(count)" to ensure non-empty selection
            selected = [label_idx for label_idx, c in enumerate(count) if c >= threshold or c == max(count)]

            # print("merged: {0} => selected: {1}".format(merged, selected))
            self.sent_labels.append(selected)

    def select_word_labels(self, threshold):
        if self.word_labels:
            raise ValueError('word_labels has already existed: {}'.format(self.word_labels))

        if not self.n_sents:
            raise ValueError('self.n_sents is None. Set self.n_sents before selecting word labels!')

        for sent_idx in range(self.n_sents):
            word_ids = [annots[sent_idx] for annots in self.word_annots]  # all workers' annots for this sentence
            merged = [int(item) for item in list(itertools.chain(*word_ids)) if item]

            distinct_ids = list(set(merged))

            id2count = dict()
            for word_id in distinct_ids:
                id2count[word_id] = merged.count(word_id)
            max_count = max(id2count.values())

            selected = [word_id for word_id, c in id2count.items() if c >= threshold or c == max_count]

            # print("merged: {0} => selected: {1}".format(merged, selected))
            self.word_labels.append(selected)


def get_annotation_fp(corpus, grain):
    """
        get fp for mturk annotations.
    :param corpus: wiki or nyt
    :return: fp
    """
    if grain == 'word':
        if lang == 'en':
            if corpus == 'wiki':
                fp = path_parser.annot_word_en_wiki
            else:
                fp = path_parser.annot_word_en_nyt
        elif lang == 'zh':
            fp = path_parser.annot_word_zh_wiki
        else:
            raise ValueError('Invalid corpus: {}'.format(corpus))
    elif grain == 'sent':
        if lang == 'en':
            if corpus == 'wiki':
                fp = path_parser.annot_sent_en_wiki
            else:
                fp = path_parser.annot_sent_en_nyt
        elif lang == 'zh':
            fp = path_parser.annot_sent_zh_wiki
        else:
            raise ValueError('Invalid corpus: {}'.format(corpus))
    else:
        raise ValueError('Invalid lang: {}'.format(lang))

    return fp


def get_annotation_n_docs(corpus):
    """
        get n_docs for mturk annotations.
    :param corpus: wiki or nyt
    :return: n_docs
    """
    if lang == 'en':
        if corpus == 'wiki':
            n_docs = 14
        else:
            n_docs = 11
    else:
        n_docs = 12
    return n_docs


def init_docs(corpus):
    n_docs = get_annotation_n_docs(corpus=corpus)
    docs = [DocRec() for _ in range(n_docs)]
    return docs


def set_sent_labels(docs, corpus, threshold, n_worker):
    sent_ks = ['Input.sent_{0}'.format(idx) for idx in range(1, config_model['max_n_sents'] + 1)]
    label_ks = ['Answer.label_{0}'.format(idx) for idx in range(1, config_model['max_n_sents'] + 1)]

    with io.open(file=get_annotation_fp(corpus, grain='sent'), encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_idx, row in enumerate(reader):
            doc_id = int(row_idx / n_worker)
            # print('row_idx % n_worker: {}'.format(row_idx % n_worker))
            if row_idx % n_worker == 0:  # first row for a doc
                # for (k, v) in row.items():
                #     print('{0}: {1}'.format(k, v))
                n_sents = int(row['n_sents'])
                docs[doc_id].set_n_sents(n_sents)
                docs[doc_id].set_fn(row['fn'])
                sents = [row[sent_k] for sent_k in sent_ks[:n_sents]]
                docs[doc_id].set_sents(sents)

            rec_labels = [row[label_k].split('|') for label_k in label_ks if label_k in row]
            # print('rec_labels: {0}'.format(rec_labels))
            docs[doc_id].add_sent_annots(rec_labels)

    for doc in docs:
        doc.select_sent_labels(threshold=threshold)


def set_word_labels(docs, corpus, source, threshold=None, n_worker=None):
    if source == 'annot':
        set_word_labels_from_annot(docs, corpus, threshold, n_worker)
    elif source == 'file':
        set_word_labels_from_file(docs, corpus)
    else:
        raise ValueError('Invalid source: {}'.format(source))


def set_word_labels_from_annot(docs, corpus, threshold, n_worker):
    label_ks = ['Answer.label_{0}'.format(idx) for idx in range(0, config_model['max_n_sents'])]

    with io.open(file=get_annotation_fp(corpus, grain='word'), encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_idx, row in enumerate(reader):
            doc_id = int(row_idx / n_worker)
            # print('row keys: {}'.format(row))
            annots = [row[label_k].split('|') for label_k in label_ks if label_k in row]
            # print('word annots: {0}'.format(annots))
            docs[doc_id].add_word_annots(annots)

    for doc in docs:
        doc.select_word_labels(threshold=threshold)


def set_word_labels_from_file(docs, corpus):
    if lang == 'en':
        if corpus == 'wiki':
            fp = path_parser.label_en_wiki
        elif corpus == 'nyt':
            fp = path_parser.label_en_nyt
        else:
            raise ValueError('Invalid corpus with EN: {}'.format(corpus))
    else:
        if corpus == 'wiki':
            fp = path_parser.label_zh_wiki
        else:
            raise ValueError('Invalid corpus with ZH: {}'.format(corpus))

    with io.open(fp, encoding='utf-8') as f:
        lines = f.readlines()

    id2pos = dict()

    for line in lines:
        items = line.rstrip('\n').split('\t')
        doc_id, sent_id, true_pos = items
        id = '.'.join((doc_id, sent_id))
        id2pos[id] = true_pos

    for doc_id, doc in enumerate(docs):
        if doc.word_labels:
            raise ValueError('word_labels has already existed: {}'.format(doc.word_labels))

        if not doc.n_sents:
            raise ValueError(
                'doc.n_sents in {}th doc is None. Set doc.n_sents before setting word labels!'.format(doc_id))

        for sent_id in range(doc.n_sents):
            id = '.'.join((str(doc_id), str(sent_id)))
            labels = [int(label) for label in id2pos[id].split('|')]
            doc.word_labels.append(labels)


def get_default_word_label_source():
    if lang == 'en':
        return 'annot'
    elif lang == 'zh':
        return 'file'
    else:
        raise ValueError('Invalid lang: {}'.format(lang))


def get_docs(corpus, threshold=3, n_worker=5, set_labels_for_words=True):
    docs = init_docs(corpus)
    set_sent_labels(docs, corpus, threshold, n_worker)
    if set_labels_for_words:
        word_label_source = get_default_word_label_source()
        set_word_labels(docs, corpus, word_label_source, threshold, n_worker)
    return docs


def get_sent_label_mat(doc):
    label_mat_shape = [config_model['max_n_sents'], n_ways]
    score_func = config_model['score_func']

    if score_func == 'tanh':
        label_mat = - np.ones(label_mat_shape, dtype=np.float32)
    elif score_func in ('sigmoid', 'softmax'):
        label_mat = np.zeros(label_mat_shape, dtype=np.float32)
    else:
        raise ValueError('Invalid score func: {}'.format(score_func))

    for sent_idx, sent_labels in enumerate(doc.sent_labels):
        for label in sent_labels:
            label_mat[sent_idx, label] = 1.0

    return label_mat


def get_word_label_tensor(doc):
    """
        for word eval, labels denote word indices of domain evidence.
    """
    label_tensor_shape = [config_model['max_n_sents'], config_model['max_n_words'], 1]
    label_tensor = np.zeros(label_tensor_shape, dtype=np.int32)  # 0 or 1

    for sent_idx, word_ids in enumerate(doc.word_labels):
        for word_id in word_ids:
            label_tensor[sent_idx, word_id, 0] = 1
            # label_tensor[sent_idx, word_id] = 1

    return label_tensor


if __name__ == '__main__':
    corpus = 'nyt'
    docs = get_docs(corpus=corpus)
    print('n_docs: {0}'.format(len(docs)))
    
    for doc in docs:
        if lang == 'zh':
            res = dataset_parser.parse_sents(fp=join(path_parser.zh_wiki, doc.fn))
        else:
            res = dataset_parser.parse_mturk(doc=doc, corpus=corpus)

        nl = sum([len(ll) for ll in doc.word_labels])

        print(str(doc))
