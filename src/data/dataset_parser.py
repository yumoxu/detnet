# -*- coding: utf-8 -*-
import io
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import itertools
import numpy as np
import re
from os import listdir
from os.path import isfile, join
from utils.res_handler import ResLoader
from utils.config_loader import logger, path_parser, config_model, doms_final, deprecated, lang
import nltk.tokenize
# from .zh_preprocessor import ZHProcessor
# import lxml.etree as ET


class DatasetParser:

    def __init__(self):
        self.doms = doms_final
        self.n_doms = len(self.doms)

        self.res_loader = ResLoader()
        self.vocab = self.res_loader.get_indexed_vocab()

        self.max_n_sents = config_model['max_n_sents']
        self.max_n_words = config_model['max_n_words']
        self.max_n_sents_des = config_model['max_n_sents_des']
        self.max_n_words_des = config_model['max_n_words_des']

        self.dp_dataset_whole = path_parser.dataset_whole
        self.dp_dataset_train = path_parser.dataset_train
        self.dp_dataset_dev = path_parser.dataset_dev
        self.dp_dataset_test = path_parser.dataset_test
        self.fp_dataset_des = path_parser.dataset_des

        self.dp_dataset_asm = path_parser.dataset_lead_sent_asm

        if lang == 'en':
            self.DES_PATTERN = '#s-doc[\s\S]*?(?=#s-headline\t1\n)'
            self.DES_PATTERN_STRIPPED = '#s-para\t1[\s\S]*?(?=#s-headline\t1\n)'
        else:
            self.DES_PATTERN = '#s-doc[\s\S]*?(?=#s-headline\n)'
            self.DES_PATTERN_STRIPPED = '#s-para\n[\s\S]*?(?=#s-headline\n)'

        self.SENT_PATTERN = '#s-sent[\s\S]*?(?=#e-sent)'  # positive lookahead
        self.SIDE_PATTERN = '(?<=#s-{0}\n)[\s\S]*?(?=\n#e-{0})'  # positive lookbehind and lookahead
        self.SENT_RAW_PATTERN = '#s-sent[\s\S]*?(?=\n)'  # positive lookahead
        self.PARA_PATTERN = '#s-para[\s\S]*?(?=#e-para)'

    @deprecated
    def parse_test_sent(self, fp, clip=True):
        """
            parse a doc as a list of sents.
            each sent is a list of word.
        """

        with io.open(fp, encoding='utf-8') as f:
            sent = f.read()
            assert sent
            # if sent:
            #     sent = sent[0]
            # else:
            #     logger.info('No sent: {0}'.format(fp))

        # the first line of a sent is #s-sent and is discarded
        sent = [word_line.split('\t')[0] for word_line in sent.split('\n')[1:]]

        if clip:  # you only index after clipped
            sent = [word for word in sent[:self.max_n_words]]

            word_id_vec = np.zeros(self.max_n_words, dtype=np.int32)

            word_id_vec[:len(sent)] = [self._index_word(word) for word in sent]

            n_words = len(sent)
            word_mask_vec = np.zeros([self.max_n_words,])
            word_mask_vec[:n_words] = [1] * n_words

            res = {
                'sent': sent,
                'word_ids': word_id_vec,
                'word_mask': word_mask_vec,
            }

            return res

        return sent

    def _index_word(self, word):
        if word not in self.vocab:
            word = 'UNK'
        return self.vocab[word]

    def _index_2d_nested_word_list(self, target, max_dims):
        """

        :param target: a list of lists.
        :param max_dims: the id_mat size.
        :return:
        """
        id_mat = np.zeros(max_dims, dtype=np.int32)

        for first_dim_id in range(len(target)):
            words = target[first_dim_id]
            id_mat[first_dim_id, :len(words)] = [self._index_word(word) for word in words]

        return id_mat

    def _index_3d_nested_word_list(self, target, max_dims):
        id_tensor = np.zeros(max_dims, dtype=np.int32)

        for first_dim_id in range(len(target)):
            id_tensor[first_dim_id] = self._index_2d_nested_word_list(target[first_dim_id], max_dims=max_dims[1:])

        return id_tensor

    def _mask_sents(self, sents, max_n_sents, max_n_words):
        # mask: sent
        n_sents = len(sents)
        sent_mask_vec = np.zeros([max_n_sents,])
        sent_mask_vec[:n_sents] = [1] * n_sents

        # mask: word
        n_words = [len(sent) for sent in sents]
        word_mask_mat = np.zeros([max_n_sents, max_n_words])
        for i, n_word in enumerate(n_words):
            word_mask_mat[i, :n_word] = [1] * n_word

        return word_mask_mat, sent_mask_vec

    def get_sents_from_paras(self, fp):
        with io.open(fp, encoding='utf-8') as f:
            content = f.read()

        para_pat = re.compile(self.PARA_PATTERN)
        sent_pat = re.compile(self.SENT_PATTERN)

        paras = re.findall(para_pat, content)
        sents = []

        if not paras:
            return sents

        for para in paras:
            ss = re.findall(sent_pat, para)
            if ss:
                sents.extend(ss)

        return sents

    def parse_sents_wo_tokenization(self, fp, in_para_only=True, clip_sents=False):
        if in_para_only:
            sents = self.get_sents_from_paras(fp)
        else:
            with io.open(fp, encoding='utf-8') as f:
                content = f.read()

            sent_pat = re.compile(self.SENT_PATTERN)
            sents = re.findall(sent_pat, content)

        if not sents:
            logger.warning('no sents found in {}'.format(fp))
            return None

        sents = [sent.split('\n')[0].split('\t')[3] for sent in sents]

        if clip_sents:
            sents = sents[:self.max_n_sents]
        return sents

    def parse_sents(self, fp, in_para_only=False, clip=True):
        """
            parse a doc as a list of sents.
            each sent is a list of word.

            in_para_only:
                Default false for classification.
                if True (for summarization use), parse only sentences between #s-para and #e-para.

        """
        if in_para_only:
            sents = self.get_sents_from_paras(fp)
        else:
            with io.open(fp, encoding='utf-8') as f:
                pattern = re.compile(self.SENT_PATTERN)
                sents = re.findall(pattern, f.read())

        if not sents:
            logger.warning('no sents found in {}'.format(fp))
            return None

        # the first line of a sent is #s-sent and is discarded
        sents = [[word_line.split('\t')[0] for word_line in sent.split('\n')[1:]] for sent in sents]
        # logger.info('sents: {0}'.format(sents))
        # logger.info('fn: {}: n_sents: {}'.format(fp, len(sents)))

        if clip:  # you only index after clipped
            sents = [[word for word in sent[:self.max_n_words]] for sent in sents[:self.max_n_sents]]
            word_id_mat = self._index_2d_nested_word_list(sents, max_dims=[self.max_n_sents, self.max_n_words])

            word_mask_mat, sent_mask_vec = self._mask_sents(sents,
                                                            max_n_sents=self.max_n_sents,
                                                            max_n_words=self.max_n_words)

            res = {
                'sents': sents,
                'word_ids': word_id_mat,
                'sent_mask': sent_mask_vec,
                'word_mask': word_mask_mat,
            }

            return res

        return sents

    def parse_mturk(self, doc, corpus, clip=True):
        """
            parse a doc as a list of sents.
            each sent is a list of word.
        """
        # if lang == 'en' and corpus == 'wiki':
        #     sents = [sent.split() for sent in doc.sents]
        # else:
        #     if lang == 'zh':
        #         tokenizer = ZHProcessor().segmentor.segment
        #     else:  # nyt
        #         tokenizer = nltk.tokenize.word_tokenize
        #     sents = [tokenizer(sent) for sent in doc.sents]
        if corpus == 'wiki':
            sents = [sent.split() for sent in doc.sents]
        else: # nyt
            sents = [nltk.tokenize.word_tokenize(sent) for sent in doc.sents]
        # logger.info('sents: {0}'.format(sents))

        if not clip:  # you only index after clipped
            return sents

        sents = [[word for word in sent[:self.max_n_words]] for sent in sents[:self.max_n_sents]]
        word_id_mat = self._index_2d_nested_word_list(sents, max_dims=[self.max_n_sents, self.max_n_words])

        word_mask_mat, sent_mask_vec = self._mask_sents(sents, max_n_sents=self.max_n_sents,
                                                        max_n_words=self.max_n_words)

        res = {
            'sents': sents,
            'word_ids': word_id_mat,
            'sent_mask': sent_mask_vec,
            'word_mask': word_mask_mat,
        }

        return res

    def get_des_sent_info(self, fp):
        des_pattern = re.compile(self.DES_PATTERN)
        des_pattern_stripped = re.compile(self.DES_PATTERN_STRIPPED)
        sent_pattern = re.compile(self.SENT_PATTERN)

        with io.open(fp, encoding='utf-8') as f:
            text = re.findall(des_pattern, f.read())[0]
            f.seek(0, 0)
            text_stripped = re.findall(des_pattern_stripped, f.read())[0]

        sents = re.findall(sent_pattern, text)
        sents_stripped = re.findall(sent_pattern, text_stripped)

        start_sent_idx = len(sents) - len(sents_stripped)

        if start_sent_idx < 0:
            logger.info('fp: {}'.format(fp))
            logger.info('sents: {}'.format(sents))
            logger.info('sents_stripped: {}'.format(sents_stripped))
            raise ValueError

        end_sent_idx = min(len(sents), config_model['max_n_sents'])
        n_sents = end_sent_idx - start_sent_idx

        return start_sent_idx, end_sent_idx, n_sents

    def parse_words(self, fp, clip=True):
        """
            parse a doc as a list of words.
        """
        res = self.parse_sents(fp, clip=clip)
        sents = res['sents'] if clip else res
        words = list(itertools.chain.from_iterable(sents))
        word_ids = [self._index_word(word) for word in words]
        return words, word_ids

    def parse_side_des(self, sent_org=True):
        # descs: if not sent_org, a 2d nested list - word list nested in dom list
        # if sent_org, a 3d nested list - word list nested in sent list nested in dom list
        descs = list()
        with io.open(self.fp_dataset_des, encoding='utf-8') as f:
            for dom in self.doms:
                pattern = re.compile(self.SIDE_PATTERN.format(dom))
                desc = re.findall(pattern, f.read())[0]

                if sent_org:
                    if lang == 'en':
                        dom_desc_sents = desc.split('. ')  # to sents
                        dom_desc = [[word for word in sent.split()[:self.max_n_words_des]]
                                    for sent in dom_desc_sents[:self.max_n_sents_des]]
                    else: # in zh, a topic is treated as a sentence
                        sent_pattern = re.compile(self.SENT_PATTERN)
                        dom_desc_sents = re.findall(sent_pattern, desc)
                        dom_desc = [[word for word in sent.split('\n')[1:self.max_n_words_des+1] if word]
                                    for sent in dom_desc_sents[:self.max_n_sents_des]]
                        # print('dom_desc: {0}'.format(dom_desc))
                else:
                    # todo: where uses no sent_org? the max length should be revised as follows
                    # max_n_words = self.max_n_words * self.max_n_sents
                    if lang == 'en':
                        dom_desc_words = desc.split()  # to words
                        dom_desc = [word for word in dom_desc_words[:self.max_n_words_des]]
                    else:
                        sent_pattern = re.compile(self.SENT_PATTERN)
                        dom_desc_sents = re.findall(sent_pattern, desc)
                        dom_desc = [word for sent in dom_desc_sents
                                    for word in sent.split('\n')[1:] if word][:self.max_n_words_des]

                descs.append(dom_desc)
                f.seek(0, 0)

        if sent_org:
            max_dims=[self.n_doms, self.max_n_sents_des, self.max_n_words_des]
            word_id_tensor = self._index_3d_nested_word_list(target=descs, max_dims=max_dims)

            word_mask_mats, sent_mask_vecs = list(), list()
            for desc in descs:
                # each desc is a list of sents
                word_mask_mat, sent_mask_vec = self._mask_sents(desc, max_n_sents=self.max_n_sents_des,
                                                                max_n_words=self.max_n_words_des)
                word_mask_mats.append(word_mask_mat)
                sent_mask_vecs.append(sent_mask_vec)

            desc_word_mask = np.stack(word_mask_mats)  # n_doms * n_sents * n_words
            desc_sent_mask = np.stack(sent_mask_vecs)  # n_doms * n_sents

            res = {
                'des': descs,
                'word_id_tensor': word_id_tensor,
                'des_word_mask_tensor': desc_word_mask,
                'des_sent_mask_mat': desc_sent_mask,
            }

            return res

        max_dims=[self.n_doms, self.max_n_words_des]
        word_id_mat = self._index_2d_nested_word_list(target=descs, max_dims=max_dims)

        return descs, word_id_mat

    def parse_lca_checking_dataset(self, integrated=False):
        des_pattern_stripped = re.compile(self.DES_PATTERN_STRIPPED)
        sent_raw_pattern = re.compile(self.SENT_RAW_PATTERN)

        def proc_func(fp):
            with io.open(fp, encoding='utf-8') as f:
                text_stripped = re.findall(des_pattern_stripped, f.read())[0]

            sent_lines = re.findall(sent_raw_pattern, text_stripped)

            if not len(sent_lines):
                logger.info('No descriptions...')
                return None

            # sents = [sent_line.split('\t')[-1] for sent_line in sent_lines[:config_model['max_n_sents']]]
            sents = [sent_line.split('\t')[-1] for sent_line in sent_lines]

            return sents

        fns = [fn for fn in listdir(self.dp_dataset_asm)
               if isfile(join(self.dp_dataset_asm, fn))]

        for fn in fns:
            src_fp = join(self.dp_dataset_asm, fn)
            sents = proc_func(src_fp)
            if integrated:
                target_fp = join(self.dp_dataset_asm, 'lca.txt')
            else:
                target_fp = join(self.dp_dataset_asm, '_'.join(('ANNOTATE', fn)))

            with io.open(target_fp, 'a', encoding='utf-8') as out_f:
                if integrated:
                    out_f.write('==========\n{0}\n'.format(fn))
                for idx, sent in enumerate(sents):
                    out_f.write('{0}\t{1}\n'.format(idx+1, sent))
            # print('fn: {0}\n{1}'.format(fn, sents))


dataset_parser = DatasetParser()

if __name__ == '__main__':
    pass
    # fp = join(dataset_parser.dp_dataset_whole, '1696648_GOV_BUS')
    # dataset_parser.parse_sents(fp)
    # dataset_parser.parse_side_top()
    # dataset_parser.parse_side_des()
    # DES_PATTERN_LAST = '#s-doc[\s\S]*?(?=#e-doc)'
    # dataset_parser.parse_lca_checking_dataset()
    # dataset_parser.parse_lca_checking_dataset(integrated=True)
    # fp = join(dataset_parser.dp_dataset_dev, '1803963_HEA')
    # start_sent_idx, end_sent_idx, n_sents = dataset_parser.get_des_sent_info(fp)
    # print(start_sent_idx, end_sent_idx, n_sents)
