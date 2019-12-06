# -*- coding: utf-8 -*-
import io
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join, isfile, dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from data.dataset_parser import DatasetParser
from utils.config_loader import deprecated, path_parser, logger, doms_final, config_model, mode, placement, lang, n_ways
from utils.mturk import get_docs, get_sent_label_mat, get_word_label_tensor
import utils.tools as tools


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
    """

    def __call__(self, numpy_dict):
        # return {'word_ids': torch.from_numpy(sample['word_ids']),
        #        'labels': torch.from_numpy(sample['labels'])}
        for (k, v) in numpy_dict.items():
            # if k == 'n_sents':
            #     continue
            if type(v) == np.ndarray:
                v = torch.from_numpy(v)
                # print('{0} size: {1}'.format(k, v.size()))

                if k.endswith('_ids'):
                    v = v.type(torch.LongTensor)  # for embedding look up

                # if k in ('des_ids', 'topic_ids', 'topic_mask', 'des_word_mask', 'des_sent_mask'):
                #     v = Variable(v, requires_grad=False)

                # if score_func == 'softmax' and k == 'labels':
                #     v = v.type(torch.LongTensor)  # for nll loss

                if placement in ('auto', 'single'):
                    v = v.cuda()

            numpy_dict[k] = v

        return numpy_dict


class DomDetDataset(Dataset):
    def __init__(self, dataset_type, transform=None, collect_doc_ids=False, doc_ids=None, in_para_only=False):
        super(DomDetDataset, self).__init__()
        self.doms = doms_final

        self.score_func = config_model['score_func']

        self.dataset_parser = DatasetParser()
        self.dataset_type = dataset_type

        assert dataset_type in ('train', 'dev', 'test', 'test_filtered')
        if mode == 'debug':
            dataset_type = 'debug'

        self.root = path_parser.dataset_type_dp[dataset_type]
        self.fns = [fn for fn in listdir(self.root) if isfile(join(self.root, fn))]
        if doc_ids:  # only one sample
            logger.info('Load only {} target docs'.format(len(doc_ids)))
            self.fns = [fn for fn in self.fns if tools.get_doc_id(fn) in doc_ids]

        self.transform = transform
        self.collect_doc_ids = collect_doc_ids
        self.in_para_only = in_para_only

    def __len__(self):
        return len(self.fns)

    def _get_labels(self, fn):
        labels = fn.split('_')[1:]
        assert self.score_func in ('sigmoid', 'tanh', 'softmax')

        if self.score_func == 'tanh':
            label_vec = - np.ones([len(self.doms)], dtype=np.float32)
        else:
            label_vec = np.zeros([len(self.doms)], dtype=np.float32)

        for label in labels:
            label_vec[self.doms.index(label)] = 1.0

        return label_vec

    def __getitem__(self, index):
        fn = self.fns[index]

        fp = join(self.root, fn)

        labels = self._get_labels(fn)
        # print('labels: {}'.format(labels.shape))

        res = self.dataset_parser.parse_sents(fp, in_para_only=self.in_para_only)

        if not res:
            return

        sample = {'labels': labels,
                  'word_ids': res['word_ids'],  # 2-dim numpy ids organized by sents
                  'sent_mask': res['sent_mask'],
                  'word_mask': res['word_mask'],
                  }

        if self.dataset_type == 'test_filtered':
            des_sent_info = self.dataset_parser.get_des_sent_info(fp)  # (start_sent_idx, end_sent_idx, n_sents)
            if des_sent_info[-1] <= 0:
                logger.info('n_sents<0, fp: {0}'.format(fp))
                assert AssertionError
            sample['des_sent_info'] = np.array(des_sent_info)

        if self.collect_doc_ids:
            sample['doc_ids'] = tools.get_doc_id(fn)

        if self.transform:
            sample = self.transform(sample)

        return sample


class SynDataset(Dataset):
    def __init__(self, transform=None):
        super(SynDataset, self).__init__()
        self.doms = doms_final
        self.score_func = config_model['score_func']
        self.dataset_parser = DatasetParser()
        self.root = path_parser.dataset_syn_docs
        self.sent_fns = [fn for fn in listdir(self.root) if fn.endswith('sent') and isfile(join(self.root, fn))]
        self.label_fns = [sent_fn.rstrip('sent') + 'label' for sent_fn in self.sent_fns]
        self.transform = transform

    def __len__(self):
        return len(self.sent_fns)

    @deprecated
    def _get_sent_labels(self, label_fn):
        with io.open(join(self.root, label_fn), encoding='utf-8') as f:
            list_of_labels = [line.strip('\n').split('_') for line in f.readlines()]

        assert self.score_func in ('sigmoid', 'tanh', 'softmax')

        list_of_label_vecs = list()
        for labels in list_of_labels:
            if self.score_func == 'tanh':
                label_vec = - np.ones([len(self.doms)], dtype=np.float32)
            else:
                label_vec = np.zeros([len(self.doms)], dtype=np.float32)

            for label in labels:
                label_vec[self.doms.index(label)] = 1.0

            list_of_label_vecs.append(label_vec)

        return np.stack(list_of_label_vecs)

    def _get_n_sents(self, label_fn):
        with io.open(join(self.root, label_fn), encoding='utf-8') as f:
            # list_of_labels = [line.strip('\n').split('_') for line in f.readlines()]  # todo: check the last \n line
            list_of_labels = [line.strip('\n').split('_') for line in f.readlines() if
                              line]  # todo: check the last \n line

        return len(list_of_labels)

    def _get_labels(self, fn):
        labels = fn.split('_')[1:-1]
        assert self.score_func in ('sigmoid', 'tanh', 'softmax')

        if self.score_func == 'tanh':
            label_vec = - np.ones([len(self.doms)], dtype=np.float32)
        else:
            label_vec = np.zeros([len(self.doms)], dtype=np.float32)

        for label in labels:
            label_vec[self.doms.index(label)] = 1.0

        return label_vec

    def _get_file_id(self, fn):
        id = int(fn.split('_')[0])
        return np.array([id])

    def __getitem__(self, index):
        sent_fn, label_fn = self.sent_fns[index], self.label_fns[index]
        sent_fp = join(self.root, sent_fn)

        labels = self._get_labels(label_fn)  # doc labels
        fids = self._get_file_id(label_fn)

        res = self.dataset_parser.parse_sents(sent_fp)

        # original way of getting n_sents
        # sent_labels = self._get_sent_labels(label_fn)
        # n_sents = sent_labels.shape[0]

        n_sents = self._get_n_sents(label_fn)

        des_sent_info = [0, n_sents, n_sents]
        des_sent_info = np.array(des_sent_info)

        sample = {'labels': labels,
                  'word_ids': res['word_ids'],  # 2-dim numpy ids organized by sents
                  'sent_mask': res['sent_mask'],
                  'word_mask': res['word_mask'],
                  'des_sent_info': des_sent_info,
                  'fids': fids,
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample


class MTurkDataset(Dataset):
    def __init__(self, corpus, threshold=3, transform=None, collect_doc_ids=False, grain='sent'):
        """
            Init MTuk dataset for sentence or word eval.

        :param corpus: 'wiki' or 'nyt'
        :param threshold: for selecting labels from mturk annotations.
        :param transform:
        :param collect_doc_ids:
        :param grain: 'sent' or 'word
        :return:
        """
        super(MTurkDataset, self).__init__()
        self.doms = doms_final
        self.score_func = config_model['score_func']
        self.dataset_parser = DatasetParser()
        self.max_n_sents = config_model['max_n_sents']
        self.max_n_words = config_model['max_n_words']
        self.corpus = corpus
        self.transform = transform
        self.collect_doc_ids = collect_doc_ids
        self.grain = grain

        set_labels_for_words = True if grain == 'word' else False
        self.docs = get_docs(corpus=corpus, threshold=threshold, set_labels_for_words=set_labels_for_words)

    def __len__(self):
        return len(self.docs)

    def _get_n_sents(self, doc):
        return np.array([doc.n_sents])

    def _get_numerical_doc_id(self, doc):
        id = int(doc.fn.split('_')[-1])
        return np.array([id])

    def _get_str_doc_id(self, doc):
        return doc.fn.split('_')[0]

    def __getitem__(self, index):
        doc = self.docs[index]

        if lang == 'zh':
            res = self.dataset_parser.parse_sents(fp=join(path_parser.zh_wiki, doc.fn))
        else:
            res = self.dataset_parser.parse_mturk(doc=doc, corpus=self.corpus)

        sent_labels = get_sent_label_mat(doc=doc)

        n_sents = self._get_n_sents(doc)  # can be derived from sent_mask. fix it when possible.
        doc_label_holder = - np.ones([n_ways], dtype=np.float32)  # doc labels are not used in MTurk

        sample = {'labels': doc_label_holder,
                  'sent_labels': sent_labels,  # n_max_sents * n_doms
                  'n_sents': n_sents,
                  'word_ids': res['word_ids'],  # 2-dim numpy ids organized by sents
                  'sent_mask': res['sent_mask'],
                  'word_mask': res['word_mask'],
                  }

        if self.grain == 'word':
            word_labels = get_word_label_tensor(doc=doc)
            sample['word_labels'] = word_labels
            sample['n_words'] = np.sum(res['word_mask'], axis=-1, dtype=np.int32)

        if self.collect_doc_ids:
            if lang == 'en':
                sample['doc_ids'] = self._get_numerical_doc_id(doc)
            else:
                sample['doc_ids'] = self._get_str_doc_id(doc)

        if self.transform:
            sample = self.transform(sample)

        return sample


class DomDetDataLoader(DataLoader):
    def __init__(self, dataset_type, collect_doc_ids=False, doc_ids=None, in_para_only=None):
        """

        :param dataset_type:
            [train | dev | test |
            mturk-[wiki|nyt]-[sent|word] |
            synthese
        :param collect_doc_ids: only used for DomDetDataset
        :return:
        """
        n_data_load_worker = config_model['n_data_load_worker']
        batch_size = config_model['batch_size']
        logger.info('load dataset: {0}'.format(dataset_type))

        if dataset_type.startswith('mturk'):
            corpus, grain = dataset_type.split('-')[1:]
            dataset = MTurkDataset(corpus=corpus, transform=ToTensor(), grain=grain)
        elif dataset_type == 'synthese':
            dataset = SynDataset(transform=ToTensor())
        else:
            dataset = DomDetDataset(dataset_type=dataset_type,
                                    transform=ToTensor(),
                                    collect_doc_ids=collect_doc_ids,
                                    doc_ids=doc_ids,
                                    in_para_only=in_para_only)

        drop_last = True if dataset_type == 'train' else False  # drop the last when training to avoid batch norm error
        shuffle = True if dataset_type == 'train' else False
        logger.info('Data shuffling: {}'.format(shuffle))

        super(DomDetDataLoader, self).__init__(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=n_data_load_worker,
                                               drop_last=drop_last)


class PriorLoader:
    def __init__(self, transform=None):
        self.desc_sent_org = config_model['desc_sent_org']
        self.doms = doms_final
        self.dataset_parser = DatasetParser()

        self.transform = transform

    def load_prior(self):
        des = self.dataset_parser.parse_side_des(sent_org=self.desc_sent_org)

        prior = {
            'des_ids': des['word_id_tensor'],
            'des_word_mask': des['des_word_mask_tensor'],
            'des_sent_mask': des['des_sent_mask_mat']
        }

        if self.transform:
            prior = self.transform(prior)

        return prior


if __name__ == '__main__':
    # data_loader = DomDetDataLoader(dataset_type='train')
    # data_loader = DomDetDataLoader(dataset_type='test_filtered')
    # data_loader = DomDetDataLoader(dataset_type='synthese')
    # 1[A|B]-[ana1|dev1|eval1+2]
    # data_loader = DomDetDataLoader(dataset_type='mat_seg-1B-ana1')
    data_loader = DomDetDataLoader(dataset_type='mturk')
    for batch_idx, batch in enumerate(data_loader):
        logger.info('batch_idx: {0}'.format(batch_idx))
        # print(batch['labels'])
        # print(batch['word_ids'].size())
        # pass
