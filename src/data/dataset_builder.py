# -*- coding: utf-8 -*-
import subprocess as sp
import copy
import io
import os
import random
import re
from os import listdir
from tqdm import tqdm
from os.path import isfile, join, dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from data.dataset_parser import DatasetParser
import data.data_pipe as pipe
from utils.config_loader import logger, path_parser, doms_final, config_model, lang
from utils.mturk import get_docs
import csv


class DatasetBuilder:
    def __init__(self):
        self.doms_final = doms_final

        # join: doc_dedu & doms_final (7 classes)
        self.dp_doc_dedu = path_parser.data_doc_dedu
        self.dom_dedu_dp = dict()
        for dom in self.doms_final:
            self.dom_dedu_dp[dom] = join(self.dp_doc_dedu, dom)

        self.dp_doc_mixed = path_parser.data_doc_mixed
        self.dps_mixed_sub = (join(self.dp_doc_mixed, '1'), join(self.dp_doc_mixed, '2'))

        self.dp_dataset_bipartite = path_parser.dataset_bipartite
        self.dps_bipartite_sub = (join(self.dp_dataset_bipartite, '1'),
                                  join(self.dp_dataset_bipartite, '2'))

        self.dp_dataset_unproc = path_parser.dataset_unproc
        self.dp_dataset_whole = path_parser.dataset_whole

        self.dp_dataset_train = path_parser.dataset_train
        self.dp_dataset_dev = path_parser.dataset_dev
        self.dp_dataset_test = path_parser.dataset_test

        self.dp_dataset_test_filtered = path_parser.dataset_test_filtered
        self.dp_dataset_dom_sent_corpora = path_parser.dataset_dom_sent_corpora
        self.dp_dataset_syn_docs = path_parser.dataset_syn_docs

        self.dp_dataset_assumption = path_parser.dataset_lead_sent_asm

        self.DIVIDE_THRESHOLD = 20000  # for building mixed data
        self.MAX_N_DOM_DOC = 5000

        assert lang in ('en', 'zh')
        assert len(self.doms_final) in (7, 8)

        if lang == 'en' and len(self.doms_final) == 7:
            self.N_TRAIN = 25562
            self.N_DEV = 3000
            self.N_TEST = 3000
        elif lang == 'en' and len(self.doms_final) == 8:
            self.N_TRAIN = 30297
            self.N_DEV = 3000
            self.N_TEST = 3000
        else:
            self.N_TRAIN = 22280
            self.N_DEV = 2000
            self.N_TEST = 2000

        if lang == 'en':
            self.DES_PATTERN = '#s-doc[\s\S]*?(?=#s-headline\t1\n)'
            self.DES_PATTERN_STRIPPED = '#s-para\t1[\s\S]*?(?=#s-headline\t1\n)'
        else:
            self.DES_PATTERN = '#s-doc[\s\S]*?(?=#s-headline\n)'
            self.DES_PATTERN_STRIPPED = '#s-para\n[\s\S]*?(?=#s-headline\n)'

        self.SENT_PATTERN = '#s-sent[\s\S]*?(?=#e-sent)'  # positive lookahead
        self.SENT_PATTERN_TAGGED = '#s-sent[\s\S]*?#e-sent'

        self.PARA_PATTERN = '#s-para[\s\S]*?(?=#e-para)'

    def build_mixed_data(self):
        """
            move files from dom_dedu_dp to dp_doc_mixed.

            divide files with a threshold and move them to two sub-dirs: 1 and 2
            to avoid system error (max #files in one folder).

            file names are changed to:
                id_label1_label2_...
        """
        done_items = list()  # [['1', 'GOV'], ['2', 'BUS', 'HEA'], ...]

        get_tgt_dp = lambda x: self.dps_mixed_sub[0] if x <= self.DIVIDE_THRESHOLD else self.dps_mixed_sub[1]
        get_old_dp = lambda fn: self.dps_mixed_sub[0] if isfile(join(self.dps_mixed_sub[0], fn)) else \
            self.dps_mixed_sub[1]

        for dom in self.doms_final:
            src_dp = self.dom_dedu_dp[dom]
            src_fps = [join(src_dp, fn) for fn in listdir(src_dp) if isfile(join(src_dp, fn))]

            for src_fp in tqdm(src_fps):
                n_done = len(done_items)
                with io.open(src_fp, encoding='utf-8') as f:
                    first_line = f.readline()

                    if lang == 'en':
                        head_tuple = first_line.split('\t')
                        head_id = head_tuple[1] if len(head_tuple) >= 2 else None
                    else:
                        head_pattern = '(?<=\<article name=")[\s\S]*?(?=">)'
                        match = re.search(re.compile(head_pattern), first_line)
                        head_id = match.group() if match else None

                    assert head_id  # docs have been cleaned, ensuring head existence

                # replace is only for zh; head id is article title in zh while is an int in en
                # _ is for splitting doc id and labels, which is what we want to keep
                # so we use - to replace /
                head_id = head_id.replace('/', '-')
                is_existing = False
                for item_id in range(len(done_items)):
                    item = done_items[item_id]

                    if item[0] == head_id:
                        is_existing = True

                        existing_labels = item[1:]
                        if dom in existing_labels:
                            break

                        old_fn = '_'.join(item)

                        old_dp = get_old_dp(fn=old_fn)
                        old_fp = join(old_dp, old_fn)

                        assert os.path.exists(old_fp)

                        item.append(dom)  # simultaneously change done_items

                        new_fn = '_'.join(item)
                        new_fp = join(old_dp, new_fn)

                        sp.call(['mv', old_fp, new_fp])
                        logger.info('{0}: add {1}'.format(head_id, dom))

                        break

                if not is_existing:
                    item = [head_id, dom]
                    tgt_fp = join(get_tgt_dp(n_done), '_'.join(item))
                    sp.call(['cp', src_fp, tgt_fp])
                    done_items.append(item)

    def label_stats(self):
        n_labels = list()
        for dp_mixed_sub in self.dps_mixed_sub:
            n_labels_sub = [len(fn.split('_')) - 1 for fn in listdir(dp_mixed_sub)
                            if isfile(join(dp_mixed_sub, fn))]
            n_labels.extend(n_labels_sub)

        one, two, three = n_labels.count(1), n_labels.count(2), n_labels.count(3)
        rest = len(n_labels) - one - two - three

        logger.info('#docs with #labels: 1 : {0}, 2 : {1}, 3 : {2}, >=4 : {3}'.format(one, two, three, rest))

    def sample(self):
        """
            after manually init dataset_bipartite with doc_mixed.
            select samples from dataset_bipartite and remove those not selected.
        """
        selected_fns_fps = []
        dict_dom_n_files = dict()
        for dom in self.doms_final:
            dict_dom_n_files[dom] = 0

        fns_fps = list()
        for dp_bipartite_sub in self.dps_bipartite_sub:
            fns_fps_sub = [(fn, join(dp_bipartite_sub, fn)) for fn in listdir(dp_bipartite_sub)
                           if isfile(join(dp_bipartite_sub, fn))]
            fns_fps.extend(fns_fps_sub)

        # fns = [fn for fn in listdir(self.dp_dataset_whole) if isfile(join(self.dp_dataset_whole, fn))]

        multi_fns_fps = [item for item in fns_fps if len(item[0].split('_')) > 2]
        single_fns_fps = [item for item in fns_fps if len(item[0].split('_')) == 2]

        logger.info('multi: {0}, single: {1}'.format(len(multi_fns_fps), len(single_fns_fps)))

        for multi_item in multi_fns_fps:
            labels = multi_item[0].split('_')[1:]
            logger.info('labels: {0}'.format(labels))
            for label in labels:
                dict_dom_n_files[label] += 1

        selected_fns_fps.extend(multi_fns_fps)

        for single_item in single_fns_fps:
            label = single_item[0].split('_')[-1]
            logger.info('label: {0}'.format(label))

            if dict_dom_n_files[label] < self.MAX_N_DOM_DOC:
                selected_fns_fps.append(single_item)
                dict_dom_n_files[label] += 1

        for item in fns_fps:
            if item not in selected_fns_fps:
                sp.call(['rm', item[1]])

        logger.info('Sampling results:')
        for k, v in dict_dom_n_files.items():
            logger.info('#samples in {0}: {1}'.format(k, v))

    def divide(self):
        """
            after creating dataset_whole by,
                [1] en: merging from dataset_bipartite;
                [2] other langs:
                        (1) merging (dataset_bipartit => dp_dataset_unproc) and
                        (2) processing (dp_dataset_unproc => dataset_whole).

            divide dataset_whole to dp_dataset_train, dp_dataset_dev and dp_dataset_test.
        """
        fns = [fn for fn in listdir(self.dp_dataset_whole) if isfile(join(self.dp_dataset_whole, fn))]
        assert len(fns) == self.N_TRAIN + self.N_DEV + self.N_TEST
        random.shuffle(fns)
        fns_train = fns[:self.N_TRAIN]
        fns_dev = fns[self.N_TRAIN: -self.N_TEST]
        fns_test = fns[-self.N_TEST:]
        zipped = ((fns_train, self.dp_dataset_train), (fns_dev, self.dp_dataset_dev), (fns_test, self.dp_dataset_test))

        for (fns, dp) in zipped:
            for fn in tqdm(fns):
                sp.call(['cp', join(self.dp_dataset_whole, fn), join(dp, fn)])
            logger.info('finished: {0}'.format(dp))

    def build_lca_checking_dataset(self, num=20):
        """
            sample [num] samples from dev set for checking the label consistency assumption (LCA).
        """
        src_fps = [join(self.dp_dataset_dev, fn) for fn in listdir(self.dp_dataset_dev)
                   if isfile(join(self.dp_dataset_dev, fn))]

        n_pick = 0

        for _ in range(len(src_fps)):
            rand_fp = random.choice(src_fps)
            if self.check_existence_of_des_sents(rand_fp):
                sp.call(['cp', rand_fp, self.dp_dataset_assumption])
                n_pick += 1
            if n_pick == num:
                break

    def check_existence_of_des_sents(self, fp):
        des_pattern = re.compile(self.DES_PATTERN)
        des_pattern_stripped = re.compile(self.DES_PATTERN_STRIPPED)
        sent_pattern = re.compile(self.SENT_PATTERN)

        with io.open(fp, encoding='utf-8') as f:
            text = re.findall(des_pattern, f.read())
            f.seek(0, 0)
            text_stripped = re.findall(des_pattern_stripped, f.read())

            if text and text_stripped:
                text = text[0]
                text_stripped = text_stripped[0]
            else:
                logger.warning('no des text found...')
                return False

            sents = re.findall(sent_pattern, text)
            sents_stripped = re.findall(sent_pattern, text_stripped)

            if not (sents and sents_stripped):
                logger.warning('no sents found...')
                return False

            start_sent_idx = len(sents) - len(sents_stripped)

            end_sent_idx = min(len(sents), config_model['max_n_sents'])
            n_sents = end_sent_idx - start_sent_idx

            if n_sents <= 0:
                return False

        return True

    def build_filtered_test_set(self):
        """
            filter test set by checking the existence of description sentences.
            files filtered are copied to path_parser.dataset_test_filtered.
        """
        logger.info('Set max_n_sents={0}'.format(config_model['max_n_sents']))
        src_fps = [join(self.dp_dataset_test, fn) for fn in listdir(self.dp_dataset_test)
                   if isfile(join(self.dp_dataset_test, fn))]

        for src_fp in tqdm(src_fps):
            if self.check_existence_of_des_sents(src_fp):
                sp.call(['cp', src_fp, path_parser.dataset_test_filtered])

    def build_cartesian_domain_sent_corpora(self):
        """
            aggregate descriptive sentences by their labels.
        """

        des_pattern_stripped = re.compile(self.DES_PATTERN_STRIPPED)

        sent_pattern = re.compile(self.SENT_PATTERN_TAGGED)  # reserve start and end

        dom_sents_dict = dict()

        src_fns = [fn for fn in listdir(self.dp_dataset_test_filtered)
                   if isfile(join(self.dp_dataset_test_filtered, fn))]

        for src_fn in tqdm(src_fns):
            label = '_'.join(src_fn.split('_')[1:])
            src_fp = join(self.dp_dataset_test_filtered, src_fn)
            with io.open(src_fp, encoding='utf-8') as src_f:
                text_stripped = re.findall(des_pattern_stripped, src_f.read())[0]
            sents_stripped = re.findall(sent_pattern, text_stripped)

            if label in dom_sents_dict:
                dom_sents_dict[label].extend(sents_stripped)
            else:
                dom_sents_dict[label] = sents_stripped

        for dom, sents in dom_sents_dict.items():
            logger.info('{0} sents: {1}'.format(dom, str(len(sents))))
            out_fp = join(self.dp_dataset_dom_sent_corpora, dom)
            with io.open(out_fp, mode='a', encoding='utf-8') as out_f:
                out_f.write('\n'.join(sents))

    def _get_a_fab_doc_with_cartesian_doms(self, use_noisy_label, doc_label_mode, shuffle=True):
        """
            generate a fabricate doc using existing label combinations.
        :param use_noisy_label: if add one noisy label to the chosen label set;
        :param doc_label_mode: use only label sets with 'multi' labels, a 'single' label or 'mixed';
        :param shuffle: if shuffle the sentences sampled.
        """

        def get_choices(cartesian_doms, sent_labels, to_string=False):
            list_of_labels = list()
            for labels in cartesian_doms:
                flag = True
                for label in labels:
                    if label not in sent_labels:
                        flag = False
                if flag:
                    list_of_labels.append(labels)
            if to_string:
                list_of_labels = ['_'.join(labels) for labels in list_of_labels]
            return list_of_labels

        cartesian_doms = [fn.split('_') for fn in listdir(self.dp_dataset_dom_sent_corpora)
                          if isfile(join(self.dp_dataset_dom_sent_corpora, fn))]

        if doc_label_mode == 'multi':
            cartesian_doms = [labels for labels in cartesian_doms if len(labels) > 1]

        if doc_label_mode == 'single':
            cartesian_doms = [labels for labels in cartesian_doms if len(labels) == 1]

        # logger.info('#cartesian_doms: {0}'.format(len(cartesian_doms)))
        doc_labels = random.choice(cartesian_doms)
        # logger.info('doc_labels: {0}'.format(doc_labels))

        doc_dict = {'doc_label': '_'.join(doc_labels),
                    'sent_dicts': list(),  # list of dics, key: 'label' and 'sent'
                    }

        sent_labels = copy.deepcopy(doc_labels)

        if 'GEN' not in sent_labels:
            sent_labels.append('GEN')

        if use_noisy_label and len(sent_labels) < len(self.doms_final):
            rest_labels = [dom for dom in self.doms_final if dom not in sent_labels]
            noisy_label = random.choice(rest_labels)
            sent_labels.append(noisy_label)

        # logger.info('sent_labels: {0}'.format(sent_labels))

        sent_label_choices = get_choices(cartesian_doms, sent_labels, to_string=True)

        n_rest_sents = config_model['max_n_sents']
        n_rest_label_choices = len(sent_label_choices)

        for label_choice in sent_label_choices:
            with io.open(join(self.dp_dataset_dom_sent_corpora, label_choice), encoding='utf-8') as f:
                dom_sents = re.findall(self.SENT_PATTERN_TAGGED, f.read())

            bound = min(len(dom_sents),
                        n_rest_sents + 1 - n_rest_label_choices,
                        int(2 * n_rest_sents / n_rest_label_choices))

            n_sample_sents = random.randint(1, bound)

            # logger.info('{0} - left: {1}, sampled: {2}'.format(label_choice, n_rest_sents, n_sample_sents))

            sampled_sents = random.sample(dom_sents, n_sample_sents)

            sampled_sents = [{'sent_label': label_choice, 'sent': sent} for sent in sampled_sents]
            doc_dict['sent_dicts'].extend(sampled_sents)

            n_rest_sents = config_model['max_n_sents'] - len(doc_dict['sent_dicts'])
            # logger.info('n_rest_sents: {0}'.format(n_rest_sents))
            n_rest_label_choices -= 1
            assert n_rest_sents >= 0

        if shuffle:
            random.shuffle(doc_dict['sent_dicts'])

        return doc_dict

    def build_fab_dataset_with_cartesian_doms(self, use_noisy_label, doc_label_mode, dataset_size=200):
        """
            generate [dataset_size] fabricate docs using existing label combinations.
        :param use_noisy_label: if add one noisy label to the chosen label set;
        :param doc_label_mode: use only label sets with 'multi' labels, a 'single' label or 'mixed';
        """
        for idx in tqdm(range(dataset_size)):
            doc_dict = self._get_a_fab_doc_with_cartesian_doms(use_noisy_label, doc_label_mode)

            basic_output_fn = list([str(idx), doc_dict['doc_label']])

            basic_output_fn.append('sent')
            out_fp_sent = join(self.dp_dataset_syn_docs, '_'.join(basic_output_fn))

            basic_output_fn[-1] = 'label'
            out_fp_label = join(self.dp_dataset_syn_docs, '_'.join(basic_output_fn))

            for sent_dict in doc_dict['sent_dicts']:
                with io.open(out_fp_sent, mode='a', encoding='utf-8') as f:
                    f.write(sent_dict['sent'] + '\n')

                with io.open(out_fp_label, mode='a', encoding='utf-8') as f:
                    f.write(sent_dict['sent_label'] + '\n')

    def is_legit_para(self, para, lower=8, upper=12):
        is_legit = False
        sent_pattern = re.compile(self.SENT_PATTERN)
        sents = re.findall(sent_pattern, para)
        if sents and lower <= len(sents) <= upper:
            is_legit = True
        return is_legit

    def check_existence_of_paras(self, fp):
        para_pattern = re.compile(self.PARA_PATTERN)
        has_legit_para = False
        with io.open(fp, encoding='utf-8') as f:
            paras = re.findall(para_pattern, f.read())
            if paras:
                for para in paras:
                    if self.is_legit_para(para):
                        has_legit_para = True

        return has_legit_para

    def sample_human_eval_fns(self, max_dom_doc=2):
        selected_fns = []
        dict_dom_n_files = dict()
        for dom in self.doms_final:
            dict_dom_n_files[dom] = 0

        root_path = path_parser.dataset_test
        target_path = path_parser.dataset_test_human_eval

        fns = [fn for fn in listdir(root_path) if
               not fn.startswith('.') and len(fn.split('_')) >= 2 and isfile(join(root_path, fn))]

        logger.info('#files: {}'.format(len(fns)))

        existing_fns = [fn for fn in listdir(target_path) if
                        not fn.startswith('.') and len(fn.split('_')) >= 2 and isfile(join(target_path, fn))]

        for fn in existing_fns:
            labels = fn.split('_')[1:]
            # logger.info('labels: {}'.format(labels))
            for label in labels:
                dict_dom_n_files[label] += 1

        for k, v in dict_dom_n_files.items():
            logger.info('Existing: {0} - {1}'.format(k, v))

        if min(dict_dom_n_files.values()) == max_dom_doc:
            logger.error('Already full!')
            raise FileExistsError

        random.shuffle(fns)
        full_status = False

        for fn in fns:
            labels = fn.split('_')[1:]
            # logger.info('labels: {0}'.format(labels))

            ahead_overflow = False
            for label in labels:
                if dict_dom_n_files[label] == max_dom_doc:
                    ahead_overflow = True
                    break

            if not ahead_overflow and self.check_existence_of_paras(fp=join(root_path, fn)):
                selected_fns.append(fn)
                for label in labels:
                    dict_dom_n_files[label] += 1

            if min(dict_dom_n_files.values()) == max_dom_doc:
                full_status = True

            if full_status:
                break

        for fn in selected_fns:
            old_path = join(root_path, fn)
            new_path = join(target_path, fn)
            sp.call(['cp', old_path, new_path])

    def get_csv_headline(self, max_n_sents=12):
        sent_ids = ['sent_{0}'.format(idx) for idx in range(1, max_n_sents + 1)]
        headline_list = ['fn', 'para', 'n_sents']
        headline_list.extend(sent_ids)
        # return ','.join(headline_list)+'\n'
        return headline_list

    def get_csv_headline_for_word_eval(self):
        headline_list = ['para', 'main_body']
        return headline_list

    def build_human_eval_dataset(self, sep=False):
        root_path = path_parser.dataset_test_human_eval
        out_path = join(root_path, 'csv')

        existing_fns = [fn for fn in listdir(out_path) if
                        not fn.startswith('.') and len(fn.split('_')) >= 2 and isfile(join(root_path, fn))]

        fns = [fn for fn in listdir(root_path) if
               not fn.startswith('.') and len(fn.split('_')) >= 2
               and isfile(join(root_path, fn)) and fn not in existing_fns]
        para_pattern = re.compile(self.PARA_PATTERN)
        sent_pattern = re.compile(self.SENT_PATTERN)

        def sample_para(fp):
            with io.open(fp, encoding='utf-8') as f:
                paras = re.findall(para_pattern, f.read())
                paras = [para for para in paras if self.is_legit_para(para)]
                assert paras
                if len(paras) > 1:
                    random.shuffle(paras)
                return paras[0]

        id2sents = dict()
        n_sents = 0
        max_n_sents = 12

        for fn in fns:
            para = sample_para(fp=join(root_path, fn))
            sents = re.findall(sent_pattern, para)
            sents = [[word_line.split('\t')[0] for word_line in sent.split('\n')[1:]] for sent in sents]
            if lang == 'en':
                sents = [' '.join(sent) for sent in sents]
            else:
                sents = [''.join(sent) for sent in sents]

            n_sents += len(sents)
            id2sents_new = {fn: sents}
            id2sents = {**id2sents, **id2sents_new}

        logger.info('#sents: {0}'.format(n_sents))

        if sep:
            for fn, sents in id2sents.items():
                out_fn = '{0}.csv'.format(fn)
                out_fp = join(out_path, out_fn)
                with open(out_fp, mode='a', encoding='utf-8') as out_f:
                    sents = [' '.join(sent) for sent in sents]
                    sents = ['\t'.join((str(idx), sent)) for idx, sent in enumerate(sents, start=1)]
                    out_f.write('\n'.join(sents))
        else:
            out_fp = join(out_path, 'one.csv')
            # with open(out_fp, mode='a', encoding='utf-8') as out_f:
            with open(out_fp, mode='a', newline='', encoding='utf_8_sig') as out_f:
                # out_f.write(headline)
                writer = csv.writer(out_f, delimiter=',')
                headline = self.get_csv_headline(max_n_sents)
                writer.writerow(headline)

                for fn, sents in id2sents.items():
                    # sents = [sent.replace('"', '""') for sent in sents]
                    # para = '"{0}"'.format('\n'.join(sents))
                    para = '\n'.join(sents)
                    # sents = ['"{0}"'.format(sent) for sent in sents]
                    logger.info("#sents: {}".format(len(sents)))

                    output_list = [fn, para, str(len(sents))]
                    output_list.extend(sents)

                    if len(sents) < max_n_sents:
                        rest = [''] * (max_n_sents - len(sents))
                        output_list.extend(rest)
                    # out_f.write(','.join(output_list)+'\n')
                    writer.writerow(output_list)

    def build_human_eval_nyt_dataset(self):
        in_fp = join(path_parser.amt, 'en_nyt.txt')
        out_fp = join(path_parser.amt, 'en_nyt.csv')

        n_sents = 0
        max_n_sents = 12
        headline = self.get_csv_headline(max_n_sents)

        doc_delimit = '=====\n'
        with open(out_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write(headline)
            with open(in_fp, encoding='utf-8') as in_f:
                docs = in_f.read().split(doc_delimit)
                for doc in docs:
                    doc = doc.rstrip(doc_delimit)
                    lines = [line.rstrip('\n') for line in doc.split('\n')]
                    fn = lines[0]
                    print(fn)
                    sents = [line.split('\t')[1] for line in lines[1:]]
                    print(sents)

                    sents = [sent.replace('"', '""') for sent in sents]
                    para = '"{0}"'.format('\n'.join(sents))
                    sents = ['"{0}"'.format(sent) for sent in sents]

                    n_sents += len(sents)

                    output_list = [fn, para, str(len(sents))]
                    output_list.extend(sents)

                    if len(sents) < max_n_sents:
                        rest = ['None'] * (max_n_sents - len(sents))
                        output_list.extend(rest)
                    out_f.write(','.join(output_list) + '\n')

                logger.info('#sents: {0}'.format(n_sents))

    def get_sent_instr(self, sent_id, sent, sent_label_ids, doms):
        if lang == 'en':
            instr_pat = '<p><label>{sent_id}.&nbsp;</label>' \
                        'The following sentence discusses <span style="color:#B22222;"><b>{dom_str_0}</b></span>: </p>' \
                        '<p>&nbsp;&nbsp;&nbsp;&nbsp;<b>{sent}</b></p>' \
                        '<p>&nbsp;&nbsp;&nbsp;&nbsp;Please tick the words relevant to <span style="color:#B22222;"><b>{dom_str_1}</b></span>. ' \
                        'Please annotate at least one word.</p>'
            sent = ' '.join(sent)
            and_pat = ' and '
            or_pat = ' or '
        else:
            instr_pat = '{sent_id}. 下面的句子讨论了 {dom_str_0}:\n' \
                        '“{sent}”\n' \
                        '请为句中涉及到 {dom_str_1} 的词汇标注1。'
            sent = ''.join(sent)
            and_pat = '和'
            or_pat = '或'

        sent_labels = ['[{0}]'.format(doms[id]) for id in sent_label_ids]

        if len(sent_labels) == 1:
            dom_str_0, dom_str_1 = sent_labels[0], sent_labels[0]
        elif len(sent_labels) == 2:
            dom_str_0, dom_str_1 = and_pat.join(sent_labels), or_pat.join(sent_labels)
        else:
            pref = ', '.join(sent_labels[:-1])
            tuple = (pref, sent_labels[-1])
            dom_str_0, dom_str_1 = and_pat.join(tuple), or_pat.join(tuple)

        return instr_pat.format(sent_id=sent_id, dom_str_0=dom_str_0, dom_str_1=dom_str_1, sent=sent)

    def get_ticks_zh(self, sent):
        # id_line = range(len(sent))
        # word_line = sent
        # return id_line, word_line
        words = list()
        for word_id, word in enumerate(sent):
            words.append('[{0}] {1}'.format(word_id, word))
        return words

    def get_ticks_en(self, sent_id, sent, vertical):
        para_pat = '<p>{}</p>'
        tab = '<span>{0}</span>'.format('&nbsp;&nbsp;&nbsp;&nbsp;')
        tick_pat = tab + '<input name="label_{sent_id}" type="checkbox" value="{word_id}" />&nbsp;{word}<br />'
        tick_pat_no_newline = '<input name="label_{sent_id}" type="checkbox" value="{word_id}" />&nbsp;{word}&nbsp;&nbsp;&nbsp;'

        ticks = ''
        for word_id, word in enumerate(sent):
            if not vertical or word_id == len(sent) - 1:
                ticks += tick_pat_no_newline.format(sent_id=sent_id, word_id=word_id, word=word)
            else:
                ticks += tick_pat.format(sent_id=sent_id, word_id=word_id, word=word)

        if not vertical:
            ticks = tab + ticks
            ticks += '<br /><br />'

        ticks = para_pat.format(ticks)
        return ticks

    def build_human_eval_dataset_for_en_words(self, corpus, threshold=3, n_max_sents=12, vertical_tick=True):
        assert lang == 'en'
        doms = ['Government and Politics', 'Lifestyle', 'Business and Commerce', 'Law and Order',
                'Physical and Mental Health', 'Military', 'General Purpose']

        root_path = path_parser.dataset_test_human_eval
        out_fp = join(root_path, 'csv', 'word_eval_{}.csv'.format(corpus))

        docs = get_docs(corpus, threshold, max_n_sents=n_max_sents, set_labels_for_words=False)
        dataset_parser = DatasetParser()

        with open(out_fp, mode='a', newline='', encoding='utf_8_sig') as out_f:
            writer = csv.writer(out_f, delimiter=',')
            headline = self.get_csv_headline_for_word_eval()
            writer.writerow(headline)

            for doc in docs:
                sents = dataset_parser.parse_mturk(doc=doc, corpus=corpus, clip=False)
                para = '<p>{0}</p>'.format(' '.join(['<p>{0}</p>'.format(' '.join(sent)) for sent in sents]))

                main_body = ''
                for sent_id, sent_info in enumerate(zip(sents, doc.labels)):
                    sent_instr = self.get_sent_instr(sent_id, sent_info[0], sent_info[1], doms=doms)
                    ticks = self.get_ticks_en(sent_id=sent_id, sent=sent_info[0], vertical=vertical_tick)
                    sent_rec = sent_instr + ticks
                    main_body += sent_rec
                output_list = [para, main_body]
                writer.writerow(output_list)

    def build_human_eval_dataset_for_zh_words(self, threshold=3, n_max_sents=12):
        assert lang == 'zh'
        doms = ['政府与政治', '生活方式', '商业与贸易', '法律与秩序', '生理与心理健康', '军事', '普适目的']

        root_path = path_parser.dataset_test_human_eval
        out_fp = join(root_path, 'csv', 'word_eval_zh.csv')

        docs = get_docs('wiki', threshold, max_n_sents=n_max_sents, set_labels_for_words=False)
        dataset_parser = DatasetParser()

        with open(out_fp, mode='a', newline='', encoding='utf_8_sig') as out_f:
            writer = csv.writer(out_f, delimiter=',')
            for doc_id, doc in enumerate(docs, start=1):
                sents = dataset_parser.parse_sents(fp=join(path_parser.zh_wiki, doc.fn), clip=False)
                writer.writerow(['这是第{0}/{1}个任务。请阅读以下段落:'.format(doc_id, len(docs))])
                para = ''.join([''.join(sent) for sent in sents])
                writer.writerow([para])
                writer.writerow(['接下来,请用1标注以下句子中与所涉及领域相关的词汇。请至少标注一个词汇。'])
                for sent_id, sent_info in enumerate(zip(sents, doc.sent_labels)):
                    sent_instr = self.get_sent_instr(sent_id, sent_info[0], sent_info[1], doms=doms)
                    writer.writerow([sent_instr])
                    word_line = self.get_ticks_zh(sent_info[0])
                    writer.writerow(word_line)
                    writer.writerow([''] * (len(sent_info[0]) - 1))
                writer.writerow([''])

    def select_pos_neg_labels_for_en_words(self, corpus):
        if lang == 'en':
            if corpus == 'wiki':
                out_fp = path_parser.label_en_wiki
            elif corpus == 'nyt':
                out_fp = path_parser.label_en_nyt
            else:
                raise ValueError('Invalid corpus with EN: {}'.format(corpus))
        else:
            if corpus == 'wiki':
                out_fp = path_parser.label_zh_wiki
            else:
                raise ValueError('Invalid corpus with ZH: {}'.format(corpus))

        grain = 'word'
        dataset_type = '-'.join(('mturk', corpus, grain))
        data_loader = pipe.DomDetDataLoader(dataset_type=dataset_type)

        records = list()
        for batch_idx, batch in enumerate(data_loader):
            word_labels = batch['word_labels'].cpu().numpy()
            n_sents = batch['n_sents'].cpu().numpy()
            n_words = batch['n_words'].cpu().numpy()

            n_doc = len(word_labels)
            logger.info('n_doc: {}'.format(n_doc))

            for doc_idx in range(n_doc):
                for sent_idx in range(n_sents[doc_idx, 0]):
                    nw = n_words[doc_idx, sent_idx]
                    labels_w = word_labels[doc_idx, sent_idx]
                    pos_w_ids = [str(w_id) for w_id in range(nw) if labels_w[w_id] == 1]
                    pos_str = '|'.join(pos_w_ids)
                    neg_w_ids = [str(w_id) for w_id in range(nw) if str(w_id) not in pos_w_ids]
                    neg_str = '|'.join(neg_w_ids)

                    record = '\t'.join((str(doc_idx), str(sent_idx), pos_str, neg_str))
                    records.append(record)

        logger.info('n_records: {}'.format(len(records)))
        with io.open(out_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write('\n'.join(records))

        logger.info('Selection has been successfully saved to: {}'.format(out_fp))

    def sample_neg_labels_for_en_words(self, corpus):
        if lang == 'en':
            if corpus == 'wiki':
                true_fp = path_parser.label_en_wiki
                pred_fp = path_parser.detnet_pred_en_wiki
                sample_fp = path_parser.sampled_label_en_wiki
            elif corpus == 'nyt':
                true_fp = path_parser.label_en_nyt
                pred_fp = path_parser.detnet_pred_en_nyt
                sample_fp = path_parser.sampled_label_en_nyt
            else:
                raise ValueError('Invalid corpus with EN: {}'.format(corpus))
        else:
            if corpus == 'wiki':
                true_fp = path_parser.label_zh_wiki
                pred_fp = path_parser.detnet_pred_zh_wiki
                sample_fp = path_parser.sampled_label_zh_wiki
            else:
                raise ValueError('Invalid corpus with ZH: {}'.format(corpus))

        with io.open(true_fp, encoding='utf-8') as true_f:
            true_lines = true_f.readlines()

        with io.open(pred_fp, encoding='utf-8') as pred_f:
            pred_lines = pred_f.readlines()

        contents = list()
        for true_l, pred_l in zip(true_lines, pred_lines):
            true_items = true_l.rstrip('\n').split('\t')
            pred_items = pred_l.rstrip('\n').split('\t')
            doc_id, sent_id, true_pos, true_negs = true_items

            if true_items[:2] != pred_items[:2]:
                raise ValueError(
                    'Incompatible doc and sentence ids between true items: {0} and pred items: {1}!'.format(true_items,
                                                                                                            pred_items))
            true_neg = true_items[-1].split('|')
            pred_neg = pred_items[-1].split('|')
            sample_scope = [neg for neg in true_neg if neg in pred_neg]

            n_samples = len(true_pos.split('|'))  # n_true_pos

            if n_samples > len(true_negs):
                raise ValueError(
                    '{0}.{1}: n_samples :{2} > n_true_negs: {3}'.format(doc_id, sent_id, n_samples, len(true_negs)))

            if len(sample_scope) >= n_samples:
                samples = random.sample(sample_scope, n_samples)
            else:
                additional_sample_scope = [neg for neg in true_neg if neg not in pred_neg]
                n_additional = n_samples - len(sample_scope)

                additional_samples = random.sample(additional_sample_scope, n_additional)
                samples = sample_scope + additional_samples
                logger.info('{0}.{1}: additional_samples: {2}'.format(doc_id, sent_id, additional_samples))

            w_str = '\t'.join((doc_id, sent_id, true_pos, '|'.join(samples)))
            contents.append(w_str)

        with io.open(sample_fp, mode='a', encoding='utf-8') as sample_f:
            sample_f.write('\n'.join(contents))

        logger.info('Negative samples has been successfully saved to: {}'.format(sample_fp))


if __name__ == '__main__':
    dataset_bd = DatasetBuilder()

    # for train/dev/test
    # dataset_bd.preprocess_zh()
    # dataset_bd.build_mixed_data()
    # dataset_bd.label_stats()
    # dataset_bd.sample()
    # dataset_bd.divide()

    # for descriptive sent
    # dataset_bd.build_lca_checking_dataset(num=4)
    # dataset_bd.build_filtered_test_set()

    # for fab sent
    # dataset_bd.build_cartesian_domain_sent_corpora()
    # dataset_bd.build_fab_dataset_with_cartesian_doms(use_noisy_label=True, doc_label_mode='mixed', dataset_size=200)

    # for human eval
    # dataset_bd.sample_human_eval_fns()
    # dataset_bd.build_human_eval_dataset()
    # dataset_bd.build_human_eval_nyt_dataset()
    # dataset_bd.build_human_eval_dataset_for_en_words(corpus='wiki', vertical_tick=False)
    dataset_bd.build_human_eval_dataset_for_zh_words()
    # dataset_bd.select_pos_neg_labels_for_en_words(corpus='wiki')
    # dataset_bd.sample_neg_labels_for_en_words(corpus='wiki')
