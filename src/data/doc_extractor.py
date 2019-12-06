# -*- coding: utf-8 -*-
import io
import re
import mmap
import subprocess as sp
from multiprocessing.dummy import Pool as ThreadPool
import os
from os import listdir
from os.path import isfile, join, dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from utils.config_loader import logger, path_parser, doms, doms_final, target_dom, target_lv, lang


class Extractor:

    def __init__(self):
        self.fp_non_ge_ids = path_parser.data_non_ge_ids
        self.fp_clean_stats = path_parser.data_clean_stats

        # join: doc & doms (8 classes)
        self.dp_doc = path_parser.data_doc
        self.dom_dp = dict()
        for dom in doms:
            self.dom_dp[dom] = os.path.join(self.dp_doc, dom)

        # join: doc_dedu & doms_final (7 classes)
        self.dp_doc_dedu = path_parser.data_doc_dedu
        self.dom_dedu_dp = dict()
        for dom in doms_final:
            self.dom_dedu_dp[dom] = os.path.join(self.dp_doc_dedu, dom)

        self.dp_proj_root = path_parser.proj_root

        self.SUBCATEGORIES = 'subcategories'
        self.PAGES = 'pages'

        if lang == 'en':
            self.START_DOC, self.END_DOC = '#s-doc\t{0}', '#e-doc\t{0}'
            self.START_SEC, self.END_SEC = 's-headline', '#e-headline'
            self.DOC_PATTERN = '#s-doc\t{0}[\s\S]*#e-doc\t{0}'
            self.SEC_PATTERN = '#s-headline[\s\S]*?(?=#s-headline)'
            self.SEC_PATTERN_LAST = '#s-headline\t{0}[\s\S]*?(?=#e-doc)'

            self.fp_wiki_corpus_head = path_parser.wiki_corpus_head
            self.fp_wiki_files = list()

            for i in range(24):
                if i < 10:
                    i = '0{0}'.format(i)
                self.fp_wiki_files.append(os.path.join(path_parser.wiki_corpus, 'wikipedia-tagged_{}.txt'.format(i)))

            # self.heads = list()
            self.id_to_head, self.head_to_id = dict(), dict()
            with io.open(self.fp_wiki_corpus_head, encoding='utf-8') as f:
                for line in f:
                    items = line.split('\t')
                    if len(items) >= 2:
                        head_id, head = int(items[0]), items[1]
                        self.id_to_head[head_id] = head
                        self.head_to_id[head] = head_id
                        # self.heads.append(head)

            logger.info('#articles: {0}'.format(len(self.id_to_head)))
        else:
            self.START_DOC, self.END_DOC = '<article name="{0}">', '</article>'
            self.START_PARAGRAPH, self.END_PARAGRAPH = '<p>', '</p>'
            self.START_HEADLINE, self.END_HEADLINE = '<h>', '</h>'
            self.DOC_PATTERN = self.START_DOC + '[\s\S]*?' + self.END_DOC
            self.fp_wiki_file = path_parser.wiki_corpus

        # multi-threads
        self.n_extraction_threads = 5  # 10
        self.n_file_threads = 5  # 6

    def get_head_id(self, head):
        if head in self.head_to_id:
            return self.head_to_id[head]
            # return self.heads.index(head) + 1 # index from 1
        else:
            logger.info('doc out of list: {0}'.format(head))
            return None

    @staticmethod
    def get_file_id(head_id):
        file_id = int(head_id / 100000)
        assert 0 <= file_id <= 23
        return file_id

    def get_doc_by_head(self, head, use_head_id=False):
        """
            get a doc as a string by its head (default) or head id (set use_head_id = True).
        """
        if lang != 'en' or use_head_id:
            head_id = head
        else:
            head_id = self.get_head_id(head)
            if not head_id:
                return None

        if lang == 'en':
            file_id = self.get_file_id(head_id)
            fp_wiki_file = self.fp_wiki_files[file_id]
        else:
            fp_wiki_file = self.fp_wiki_file

        with io.open(fp_wiki_file, encoding='utf-8') as f:
            data = mmap.mmap(f.fileno(), 0, access=mmap.PROT_READ)
            pattern = re.compile(self.DOC_PATTERN.format(head_id).encode('utf-8'))
            match = re.search(pattern, data)
            if match:
                logger.info('succeeded to extract doc: {0}'.format(head_id))
                return match.group().decode('utf-8')
            else:
                logger.info('failed to extract doc: {0}'.format(head_id))
                return None

    def save_doc(self, doc, fp):
        assert doc
        if not os.path.exists(fp):
            with io.open(fp, mode='a', encoding='utf-8') as f:
                f.write(doc)

    def dump_doc(self, lv, prefix):
        if type(prefix) is not list:
            prefix = [prefix]
        prefix_str = '-'.join(prefix).replace('/', '_')

        fns, fps = list(), list()

        if lang == 'en' and lv == 'subsubtopic' and prefix[0] == 'Sports':
            # in subsubtopic - SPO, there are too many files
            # we put them into a separate dir and divide it into 2 subdirs
            src_root = join(path_parser.lv_dp[lv], 'Sports')
            src_roots = [join(src_root, '1'), join(src_root, '2')]

            for src_root in src_roots:
                for fn in listdir(src_root):
                    fp = join(src_root, fn)
                    if fn.startswith(prefix_str) and fn.endswith(self.PAGES) and isfile(fp):
                        fns.append(fn)
                        fps.append(fp)
        else:
            if lang == 'zh':
                # zh relational data is organized by lv - dom
                src_root = join(path_parser.lv_dp[lv], prefix[0])
            else:
                # the other en relational data is organized by lv only
                src_root = path_parser.lv_dp[lv]

            for fn in listdir(src_root):
                    fp = join(src_root, fn)
                    if fn.startswith(prefix_str) and fn.endswith(self.PAGES) and isfile(fp):
                        fns.append(fn)
                        fps.append(fp)

        logger.info('extract docs for: {0}'.format(', '.join(fns)))

        file_pool = ThreadPool(self.n_file_threads)
        ext_pool = ThreadPool(self.n_extraction_threads)

        def _proc_file(fn, fp):
            def _get_and_save(concat_head):
                no_slash_concat_head = concat_head.replace('/', '_')
                out_fp = os.path.join(self.dom_dp[dom], lv, no_slash_concat_head)

                if os.path.exists(out_fp):
                    logger.info('file was pre-extracted...')
                    return 1

                split_head = concat_head.replace('_', ' ')
                doc = self.get_doc_by_head(split_head)

                if doc:
                    self.save_doc(doc=doc, fp=out_fp)
                    return 1
                else:
                    return 0

            dom = fn.split('-')[0]
            # fp = join(path_parser.lv_dp[lv], fn)
            with io.open(fp, encoding='utf-8') as f:
                concat_heads = [head.rstrip('\n') for head in f.readlines()]
                results = ext_pool.map(_get_and_save, concat_heads)
            return len(concat_heads), sum(results)

        # multi-threads on files
        results = file_pool.starmap(_proc_file, zip(fns, fps))
        ext_pool.close()
        ext_pool.join()
        file_pool.close()
        file_pool.join()

        total, success = zip(*results)  # unzip
        n_files = sum(total)
        n_success = sum(success)
        ratio = float(n_success) / n_files
        logger.info('extraction stats: all: {0}, success: {1}, ratio: {2:.2f}'.format(n_files, n_success, ratio))

    def clean_doc(self, dom):
        """
            after manually make a dom folder plain,
            this func cleans docs by:
            (1) remove ill-formed files;
            (2) remove duplicated files.
        """

        assert dom in doms_final
        head_ids = list()
        n_du = 0
        n_ill = 0

        fps = [os.path.join(self.dom_dedu_dp[dom], fn)
               for fn in listdir(self.dom_dedu_dp[dom])
               if isfile(join(self.dom_dedu_dp[dom], fn))]

        logger.info('#original file: {0}'.format(len(fps)))

        for fp in fps:
            with io.open(fp, encoding='utf-8') as f:
                first_line = f.readline()
                if lang == 'en':
                    doc_start = '#s-doc'
                    head_tuple = first_line.split('\t')
                    head_id = head_tuple[1] if len(head_tuple)>=2 else None
                else:
                    doc_start = '<article'
                    head_pattern = '(?<=\<article name=")[\s\S]*?(?=">)'
                    match = re.search(re.compile(head_pattern), first_line)
                    head_id = match.group() if match else None

                is_well_formed = first_line.startswith(doc_start) and head_id

                if not is_well_formed:
                    n_ill += 1
                    sp.call(['rm', fp])
                    logger.info('ill-formed...')
                    continue

                if head_id in head_ids:
                    logger.info('duplicated...')
                    n_du += 1
                    sp.call(['rm', fp])
                else:
                    head_ids.append(head_id)

        n_filtered = len(head_ids)
        n_all = n_du + n_ill + n_filtered

        id_out_fp = os.path.join(self.dp_doc_dedu, dom + '_ids.txt')

        with io.open(id_out_fp, 'a', encoding='utf-8') as id_f:
            id_f.write('\n'.join(head_ids))

        with io.open(self.fp_clean_stats, 'a', encoding='utf-8') as stats_f:
            stats_f.write('{0}\tall: {1}\tdu: {2}\till: {3}\tleft: {4}\n'.format(dom, n_all, n_du, n_ill, n_filtered))

    def check_cross_du(self):
        for target_dom in doms_final:
            doms_final_copy = list(doms_final)
            doms_final_copy.remove(target_dom)
            other_dom_ids = list()

            for dom in doms_final_copy:
                id_fp = os.path.join(self.dp_doc_dedu, dom + '_ids.txt')
                with io.open(id_fp, encoding='utf-8') as id_f:
                    ids = [l.rstrip('\n') for l in id_f.readlines() if l]
                    other_dom_ids.extend(ids)

            target_id_fp = os.path.join(self.dp_doc_dedu, target_dom + '_ids.txt')

            with io.open(target_id_fp, encoding='utf-8') as target_id_f:
                target_ids = [l.rstrip('\n') for l in target_id_f.readlines() if l]
                du_ids = [target_id for target_id in target_ids if target_id in other_dom_ids]
                ratio = float(len(du_ids)) / len(target_ids)
                logger.info('{0}: {1}/{2}, {3:.2f}'.format(target_dom, len(du_ids), len(target_ids), ratio))

    def check_cross_du_2(self):
        dom_ids_list = list()
        unique_ids = list()
        for dom in doms_final:
            id_fp = os.path.join(self.dp_doc_dedu, dom + '_ids.txt')
            with io.open(id_fp, encoding='utf-8') as id_f:
                dom_ids = [l.rstrip('\n') for l in id_f.readlines() if l]
                dom_ids_list.append(dom_ids)
                unique_ids.extend([dom_id for dom_id in dom_ids if dom_id not in unique_ids])

        n_label_list = [0] * len(doms_final)
        for unique_id in unique_ids:
            n_labels = 0
            for dom_ids in dom_ids_list:
                if unique_id in dom_ids:
                    n_labels += 1
            n_label_list[n_labels] += 1

        n_label_ratio = ['{0:.2f}'.format(float(n_label) / sum(n_label_list)) for n_label in n_label_list]
        logger.info('#labels: {}'.format(n_label_list))
        logger.info('ratio of #labels: {}'.format(n_label_ratio))

if __name__ == '__main__':
    ext = Extractor()

    # test
    # dom_head_ids = extractor.get_dom_head_ids()
    # print(dom_head_ids)
    # doc = extractor.get_doc(dom_head_ids[0])
    # extractor.save_doc(doc, 0)

    # for General (deprecated)
    # ext.build_non_ge_ids_file()
    # ext.get_non_ge_ids()
    # ext.get_ge_docs()

    # ext.dump_doc(lv=target_lv, prefix=target_dom)  # GEN has no dom page
    # ext.clean_doc(dom=target_dom)
    # ext.check_cross_du()
    # ext.check_cross_du_2()

    # data = '<article name="1MDC">'
    # head_id_pattern = '(?<=\<article name=")[\s\S]*?(?=">)'
    # match = re.search(re.compile(head_id_pattern), data)
    # head_id = match.group()
    # print(head_id)
